# Hướng Dẫn Chạy Và Cấu Hình

Tài liệu này mô tả cách chạy dự án và cấu hình biến môi trường cho hệ thống chatbot RAG.

## 1) Yêu cầu

- Python 3.11+
- Docker + Docker Compose
- API key cho provider chat (Groq/DeepSeek/Qwen)
- (Tuỳ chọn) API key Qwen nếu dùng embedding qua API

## 2) Cấu hình môi trường

1. Tạo file `.env` từ mẫu:

```bash
cp .env.example .env
```

2. Cập nhật các biến quan trọng trong `.env`:

- `DATABASE_URL`: Chuỗi kết nối database
- `QDRANT_URL`: URL Qdrant (ví dụ `http://qdrant:6333` khi chạy Docker Compose)
- `REDIS_URL`: URL Redis
- `SITE_BASE_URL`: Base URL để dựng link nguồn bài viết
- `DEFAULT_CHAT_PROVIDER`: `groq` | `deepseek` | `qwen`
- `CHAT_FALLBACK_ORDER`: Thứ tự fallback, ví dụ `qwen,deepseek,groq`
- `EMBEDDING_PROVIDER`: `local` hoặc `qwen_api`
- `LOCAL_EMBEDDING_MODEL`: tên model sentence-transformers khi dùng local embedding
- `EMBEDDING_DIMENSION`: số chiều vector embedding

> Lưu ý:
>
> - Nếu `EMBEDDING_PROVIDER=qwen_api` thì bắt buộc có `QWEN_API_KEY`.
> - Nếu chỉ dùng Groq cho chat, embedding nên để `local`.

## 3) Chạy bằng Docker Compose (khuyến nghị)

```bash
docker compose up --build
```

Service mặc định:

- App: `http://localhost:8000`
- Postgres: `localhost:5432`
- Qdrant: `http://localhost:6333`
- Redis: `localhost:6379`

Health check API:

```bash
curl http://localhost:8000/healthz
```

## 4) Chạy local (không dùng Docker cho app)

1. Cài dependency:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Đảm bảo Postgres/Qdrant/Redis đang chạy (local hoặc remote) và `.env` đã đúng.

3. Chạy app:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 5) Endpoint chính

- `POST /chat`
- `GET /sessions?user_id=<user_id>`
- `GET /sessions/{session_id}/messages`
- `DELETE /sessions/{session_id}`
- `POST /admin/reindex`
- `GET /admin/index-status`

## 6) Gợi ý cấu hình nhanh theo kịch bản

### Chỉ dùng Groq cho chat

- `DEFAULT_CHAT_PROVIDER=groq`
- `CHAT_FALLBACK_ORDER=groq`
- `EMBEDDING_PROVIDER=local`

### Dùng Qwen cho cả chat và embedding API

- `DEFAULT_CHAT_PROVIDER=qwen`
- `CHAT_FALLBACK_ORDER=qwen,deepseek,groq`
- `EMBEDDING_PROVIDER=qwen_api`
- Cần `QWEN_API_KEY`

## 7) Lưu ý bảo mật

- Không commit file `.env` thật vào Git.
- Chỉ commit `.env.example`.
- API keys cần lưu qua secret manager hoặc CI/CD secrets ở môi trường production.
