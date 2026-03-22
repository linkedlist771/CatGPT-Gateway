"""
Image handler — detects, extracts, and downloads generated images.

When ChatGPT generates an image via DALL-E, the response contains:
- An <img> tag with the image URL (hosted on openai.com)
- A "Image created" text indicator
- An image title/alt text (description of what was generated)

This module:
1. Detects if the last assistant message contains generated images
2. Extracts image URLs and metadata
3. Downloads images to local disk
4. Returns ImageInfo objects with URLs and local paths
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from patchright.async_api import Page

from src.config import Config
from src.selectors import Selectors
from src.chatgpt.models import ImageInfo
from src.log import setup_logging

log = setup_logging("image_handler")


async def detect_images_in_response(page: Page) -> list[dict]:
    """
    Check the last conversation turn for generated images.

    ChatGPT DALL-E image responses do NOT use data-message-author-role.
    Instead, images appear inside an article turn with:
    - img[alt="Generated image"]
    - div[id^="image-"] containers
    - src from chatgpt.com/backend-api/estuary/content

    Returns a list of dicts: [{url, alt, title}, ...] or empty list.
    """
    result = await page.evaluate("""
        () => {
            const buildTitle = (root) => {
                let title = '';

                for (const btn of root.querySelectorAll('button')) {
                    const text = (btn.innerText || '').trim();
                    const bulletIdx = text.indexOf('•');
                    if (bulletIdx > -1 && bulletIdx < text.length - 1) {
                        title = text.substring(bulletIdx + 1).trim();
                        break;
                    }
                }

                if (!title) {
                    const lines = (root.innerText || '')
                        .split('\\n')
                        .map((line) => line.trim())
                        .filter(Boolean);
                    const ignored = new Set([
                        'Image created',
                        'Creating image',
                        'Share',
                        'Edit',
                        'Save',
                        'Download',
                    ]);
                    for (const line of lines) {
                        if (line.length > 5 && line.length < 200 && !ignored.has(line)) {
                            title = line;
                            break;
                        }
                    }
                }

                return title;
            };

            const collectImages = (root) => {
                const results = [];
                const seen = new Set();
                const title = buildTitle(root);
                const hasDownloadControl = !!root.querySelector(
                    'button[aria-label*="Download"], a[aria-label*="Download"], a[download]'
                );

                for (const img of root.querySelectorAll('img')) {
                    const src = img.currentSrc || img.src || '';
                    const alt = img.alt || '';
                    const ariaHidden = (img.getAttribute('aria-hidden') || '').toLowerCase();
                    const rect = img.getBoundingClientRect();
                    const w = img.naturalWidth || img.width || rect.width || 0;
                    const h = img.naturalHeight || img.height || rect.height || 0;
                    const visible = rect.width > 24 && rect.height > 24;
                    const largeEnough = w >= 180 || h >= 180 || rect.width >= 180 || rect.height >= 180;
                    const looksLikeUiIcon =
                        alt.toLowerCase().includes('avatar') ||
                        alt.toLowerCase().includes('icon') ||
                        src.includes('avatar') ||
                        src.startsWith('data:image/svg');

                    if (!visible || !largeEnough || looksLikeUiIcon || ariaHidden === 'true') {
                        continue;
                    }

                    let normalizedSrc = src;
                    if (src) {
                        try {
                            const url = new URL(src);
                            normalizedSrc = `${url.origin}${url.pathname}|${url.searchParams.get('id') || ''}`;
                        } catch (_) {
                            normalizedSrc = src.split('?')[0];
                        }
                    }

                    const key = normalizedSrc || `${Math.round(rect.width)}x${Math.round(rect.height)}`;
                    if (seen.has(key)) continue;
                    seen.add(key);

                    if (
                        hasDownloadControl ||
                        src.startsWith('blob:') ||
                        src.includes('backend-api') ||
                        src.includes('openai') ||
                        src.includes('oaidalle') ||
                        src.includes('files.oaiusercontent.com') ||
                        alt.toLowerCase().includes('generated') ||
                        (root.innerText || '').includes('Image created')
                    ) {
                        results.push({ url: src, alt, title });
                        continue;
                    }

                    results.push({ url: src, alt, title });
                }

                return results;
            };

            const agentTurns = Array.from(document.querySelectorAll('.agent-turn'));
            if (agentTurns.length > 0) {
                const fromAgentTurn = collectImages(agentTurns[agentTurns.length - 1]);
                if (fromAgentTurn.length > 0) {
                    return fromAgentTurn;
                }
            }

            const articles = Array.from(document.querySelectorAll('article'));
            if (articles.length === 0) {
                return collectImages(document.body);
            }

            const isAssistantLikeTurn = (article) => {
                if (article.querySelector('[data-message-author-role="assistant"], .agent-turn')) {
                    return true;
                }

                const text = (article.innerText || '').toLowerCase();
                if (
                    text.includes('image created') ||
                    text.includes('creating image') ||
                    text.includes('generated image')
                ) {
                    return true;
                }

                return !!article.querySelector(
                    'button[aria-label*="Download"], a[aria-label*="Download"], a[download]'
                );
            };

            let lastTurn = null;
            for (let i = articles.length - 1; i >= 0; i--) {
                if (isAssistantLikeTurn(articles[i])) {
                    lastTurn = articles[i];
                    break;
                }
            }
            if (!lastTurn) lastTurn = articles[articles.length - 1];
            const fromLastTurn = collectImages(lastTurn);
            if (fromLastTurn.length > 0) {
                return fromLastTurn;
            }

            return collectImages(document.body);
        }
    """)

    if result:
        log.info(f"Detected {len(result)} generated image(s) in response")
        for i, img in enumerate(result):
            log.debug(f"  Image {i+1}: alt='{img.get('alt', '')[:50]}', url={img.get('url', '')[:80]}...")
    else:
        log.debug("No generated images detected in response")

    return result or []


async def download_image(page: Page, url: str, filename_hint: str = "") -> str:
    """
    Download an image from a URL using the browser's fetch API.

    Uses the browser context so cookies/auth are preserved (required
    for OpenAI-hosted images that may need authentication).

    Returns the local file path.
    """
    Config.ensure_dirs()

    # Generate a filename from the URL or hint
    if filename_hint:
        # Clean the hint for use as filename
        safe_name = re.sub(r'[^\w\s-]', '', filename_hint)[:60].strip()
        safe_name = re.sub(r'\s+', '_', safe_name)
    else:
        # Use hash of URL as filename
        safe_name = hashlib.md5(url.encode()).hexdigest()[:12]

    # Add timestamp to avoid collisions
    ts = int(time.time())
    filename = f"{safe_name}_{ts}.png"
    local_path = Config.IMAGES_DIR / filename

    log.info(f"Downloading image to {local_path}...")

    try:
        # Use browser's fetch to download (preserves auth cookies)
        image_data = await page.evaluate("""
            async (url) => {
                try {
                    const response = await fetch(url);
                    if (!response.ok) return null;
                    const blob = await response.blob();
                    const reader = new FileReader();
                    return new Promise((resolve) => {
                        reader.onloadend = () => resolve(reader.result);
                        reader.readAsDataURL(blob);
                    });
                } catch (e) {
                    return null;
                }
            }
        """, url)

        if image_data and image_data.startswith("data:"):
            # Strip the data URL prefix to get raw base64
            import base64
            header, b64data = image_data.split(",", 1)

            # Detect actual format from MIME type
            if "png" in header:
                ext = ".png"
            elif "jpeg" in header or "jpg" in header:
                ext = ".jpg"
            elif "webp" in header:
                ext = ".webp"
            else:
                ext = ".png"

            # Update filename with correct extension
            filename = f"{safe_name}_{ts}{ext}"
            local_path = Config.IMAGES_DIR / filename

            raw_bytes = base64.b64decode(b64data)
            local_path.write_bytes(raw_bytes)

            size_kb = len(raw_bytes) / 1024
            log.info(f"Image saved: {local_path} ({size_kb:.1f} KB)")
            return str(local_path)

        else:
            log.warning("Failed to fetch image data via browser")

    except Exception as e:
        log.error(f"Image download failed: {e}", exc_info=True)

    # Fallback: try using the page to download via navigation
    # (less reliable but works for some cases)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(local_path))
        log.info(f"Image saved via urllib: {local_path}")
        return str(local_path)
    except Exception as e2:
        log.error(f"Fallback download also failed: {e2}")

    return ""


async def extract_images_from_response(page: Page) -> list[ImageInfo]:
    """
    Full pipeline: detect images in the last response, download them,
    and return ImageInfo objects with both URLs and local paths.
    """
    raw_images = await detect_images_in_response(page)

    if not raw_images:
        return []

    image_infos = []
    for img_data in raw_images:
        url = img_data.get("url", "")
        alt = img_data.get("alt", "")
        title = img_data.get("title", "")

        # Download the image
        hint = alt or title or "chatgpt_image"
        local_path = await download_image(page, url, filename_hint=hint)

        image_infos.append(ImageInfo(
            url=url,
            alt=alt,
            local_path=local_path,
            prompt_title=title,
        ))

    log.info(f"Processed {len(image_infos)} image(s)")
    return image_infos
