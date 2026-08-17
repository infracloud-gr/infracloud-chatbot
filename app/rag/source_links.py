def build_article_url(base_url: str, category_slug: str, article_slug: str) -> str:
    return f"{base_url.rstrip('/')}/{category_slug}/{article_slug}"


def build_source(metadata: dict, base_url: str) -> dict:
    return {
        "article_id": metadata["article_id"],
        "title": metadata.get("title", "Untitled"),
        "heading_path": metadata.get("heading_path"),
        "url": build_article_url(base_url, metadata["category_slug"], metadata["article_slug"]),
        "thumbnail_url": metadata.get("thumbnail_url"),
        "description": metadata.get("meta_description"),
    }
