# Thiết Kế Hệ Thống RAG Chatbot

## 1. Tổng Quan

Hệ thống chatbot hỏi-đáp dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**, sử dụng **LangChain** làm framework điều phối. Dữ liệu nguồn **không còn lưu dưới dạng file tĩnh trên đĩa (hardcode)**, mà được lưu trong **database theo mô hình dạng blog/category** (ví dụ bảng `articles` thuộc về một `category`), trong đó **nội dung bài viết vẫn ở định dạng Markdown** (lưu trong cột kiểu `TEXT`). Nội dung này được chunk theo **đoạn văn (paragraph chunking)** và lưu trữ vector trong **Qdrant** (chạy dạng server, dữ liệu persistent — không cần rebuild lại toàn bộ khi restart hoặc khi chỉ có một bài viết thay đổi).

### Mục tiêu chức năng chính

1. Trả lời câu hỏi dựa trên knowledge base — nội dung được quản trị dưới dạng bài viết (blog/article) theo category, lưu trong database, nội dung ở định dạng Markdown.
2. Hỗ trợ nhiều LLM backend: **DeepSeek**, **Groq**, **Qwen** — quản lý thống nhất qua LangChain, có thể chuyển đổi linh hoạt (fallback / routing).
3. Lưu trữ **lịch sử hội thoại** (session + message) để người dùng xem lại.
4. Có cơ chế **rate limit** để chống lạm dụng API.
5. **Tự động đồng bộ vector database** mỗi khi dữ liệu (bài viết) trong database thay đổi (thêm mới / cập nhật / xoá / đổi category).
6. **Đính kèm link bài viết chi tiết** trong câu trả lời, dựng từ `base_url` + đường dẫn category + đường dẫn bài viết, giúp người dùng bấm vào xem trọn vẹn nguồn tham khảo.

### Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.11+ |
| Orchestration | LangChain (LCEL - LangChain Expression Language) |
| Vector Database | Qdrant (chạy dạng server — Docker/Qdrant Cloud, dữ liệu persistent, hỗ trợ upsert/delete theo point ID không cần rebuild toàn bộ) |
| LLM Providers | DeepSeek API, Groq API, Qwen API (qua `langchain_deepseek`, `langchain_groq`, hoặc `ChatOpenAI`-compatible wrapper cho các API tương thích OpenAI schema) |
| Embedding model | **Qwen embedding API** (dòng `Qwen3-Embedding`, qua Alibaba DashScope/Model Studio, endpoint tương thích OpenAI — dùng chung hệ sinh thái với LLM Qwen đang dùng để chat) làm lựa chọn chính nếu có key Qwen; fallback cục bộ bằng `sentence-transformers` khi cần chạy offline hoặc giảm chi phí gọi API. Lưu ý: DeepSeek và Groq hiện **không cung cấp endpoint embedding**, chỉ có API chat/completion — **nếu hệ thống chỉ dùng Groq (không có Qwen/DeepSeek), embedding bắt buộc phải chạy cục bộ bằng `sentence-transformers`**, đây không còn là fallback tuỳ chọn mà là giải pháp chính duy nhất. |
| Backend API | FastAPI |
| Lưu nội dung knowledge base | PostgreSQL/MySQL (bảng `categories`, `articles`) qua SQLAlchemy — nội dung bài viết ở định dạng Markdown |
| Sinh link bài viết trong câu trả lời | `base_url` cấu hình qua biến môi trường + `slug` của `category` và `article`, ghép thành URL chi tiết bài viết |
| Lưu chat history | SQLite (dev) / PostgreSQL (production) qua SQLAlchemy |
| Rate limiting | `slowapi` (dựa trên `limits`) hoặc Redis + token bucket tự viết |
| Đồng bộ thay đổi dữ liệu | SQLAlchemy event hook (`after_insert`/`after_update`/`after_delete`) hoặc outbox table + background worker, kết hợp APScheduler cho polling định kỳ |
| Quản lý version dữ liệu | `updated_at` (tín hiệu chính, DB tự cập nhật, không tốn chi phí tính toán) + `content_hash` dùng hash nhanh phi mật mã (`xxhash`/CRC32 thay vì SHA-256) làm lớp kiểm tra phụ, tính bất đồng bộ trong background |
| Đóng gói & triển khai | Docker (multi-stage build cho app) + Docker Compose (orchestrate app, Postgres, Qdrant, Redis) |
| CI/CD | GitHub Actions — build image và tự động đẩy lên Docker Hub dựa trên secret cấu hình trên repo GitHub |

---

## 2. Kiến Trúc Tổng Thể

```
                ┌─────────────────────────┐
                │   Knowledge Base DB      │
                │  (categories, articles   │
                │   — content: Markdown)   │
                └────────────┬─────────────┘
                             │ event hook / outbox / polling
                             ▼
                ┌─────────────────────────┐
                │   Ingestion Pipeline     │
                │  (Loader → Chunker →     │
                │   Embedder → Qdrant)     │
                └────────────┬─────────────┘
                             ▼
                ┌─────────────────────────┐
                │   Qdrant Vector Store    │
                │   (persistent, server)   │
                └────────────┬─────────────┘
                             │ retriever.invoke(query)
                             ▼
      ┌───────────────┐   RAG Chain    ┌───────────────────┐
      │  FastAPI API   │──────────────▶│  LangChain Router   │
      │  (/chat, /...) │◀──────────────│ DeepSeek/Groq/Qwen  │
      └───────┬────────┘   response    └───────────────────┘
              │
              ▼
      ┌───────────────┐        ┌───────────────┐
      │  Rate Limiter  │        │  Chat History  │
      │ (per user/IP)  │        │  DB (SQLite/PG)│
      └───────────────┘        └───────────────┘
```

---

## 3. Xử Lý Dữ Liệu (Ingestion Pipeline)

### 3.1. Mô Hình Dữ Liệu Nguồn (Database)

Knowledge base được quản trị như một hệ thống blog đơn giản:

- **Bảng `categories`**: `category_id`, `name`, `slug`, `description`.
- **Bảng `articles`**: `article_id`, `category_id` (FK), `title`, `slug` (dùng để dựng URL chi tiết bài viết), `content` (kiểu `TEXT`/`LONGTEXT`, nội dung ở định dạng **Markdown**), `status` (`draft`/`published`), `content_hash` (hash nhanh phi mật mã — `xxhash`/CRC32 — của `content`, tính bất đồng bộ, dùng làm lớp kiểm tra phụ), `created_at`, `updated_at` (tín hiệu chính, tự động cập nhật bởi DB/ORM mỗi lần ghi, không tốn chi phí tính toán).
- **Trường SEO/preview cho link** (cho phép hiển thị rich preview dạng thumbnail/title/description khi trả link trong câu trả lời): `thumbnail_url` (ảnh đại diện bài viết), `meta_description`/`excerpt` (đoạn mô tả ngắn, nếu không nhập thì fallback lấy N ký tự đầu của `content` sau khi strip Markdown).
- Chỉ những bài viết có `status = published` mới được đưa vào knowledge base / index.
- **Đường dẫn chi tiết bài viết** được dựng theo quy tắc: `{base_url}/{category.slug}/{article.slug}` (`base_url` lấy từ config, ví dụ biến môi trường `SITE_BASE_URL`).

### 3.2. Loader

- Thay vì đọc file từ đĩa, Loader truy vấn trực tiếp database (qua SQLAlchemy) để lấy các bài viết `published`.
- Có thể viết một `Document Loader` tuỳ chỉnh kế thừa `BaseLoader` của LangChain, load từng `article` thành một `Document` với:
  - `page_content`: nội dung Markdown (`article.content`).
  - `metadata`: `article_id`, `category_id`, `category_name`, `category_slug`, `article_slug`, `title`, `thumbnail_url`, `meta_description`, `content_hash`, `updated_at`.
- Có thể load toàn bộ (full sync khi khởi động) hoặc load theo danh sách `article_id` cụ thể (incremental sync khi có thay đổi).

### 3.3. Paragraph Chunking

- Không dùng `RecursiveCharacterTextSplitter` theo ký tự đơn thuần, mà **tách theo đoạn văn** dựa trên dấu xuống dòng kép (`\n\n`) — tương ứng ranh giới đoạn văn tự nhiên trong nội dung Markdown lấy từ cột `content`.
- Có thể dùng `MarkdownHeaderTextSplitter` trước để giữ ngữ cảnh heading (H1/H2/H3) trong bài viết, sau đó áp dụng tách đoạn văn bên trong từng section để mỗi chunk vẫn mang theo thông tin heading cha (giúp retrieval chính xác hơn).
- Mỗi đoạn văn quá ngắn (< ngưỡng ký tự) sẽ được gộp với đoạn liền kề để tránh chunk vô nghĩa; đoạn quá dài sẽ được chia nhỏ thêm theo câu.
- Metadata mỗi chunk: `article_id`, `category_id`, `category_name`, `category_slug`, `article_slug`, `title`, `thumbnail_url`, `meta_description`, `heading_path`, `chunk_index`, `content_hash`.

### 3.4. Embedding & Lưu Vector

- **Embedding model dùng API `Qwen3-Embedding`** (qua LangChain, cấu hình `DashScopeEmbeddings` hoặc `OpenAIEmbeddings` trỏ `base_url` sang endpoint DashScope tương thích OpenAI) — tận dụng cùng provider Qwen đang dùng cho chat, giảm số lượng nhà cung cấp cần quản lý key/billing riêng.
  - **Lưu ý quan trọng**: DeepSeek và Groq **không có API embedding**, chỉ phục vụ chat/completion, nên không thể dùng 2 provider này cho bước embedding — dù model chat vẫn có thể chọn linh hoạt giữa cả 3 (mục 4).
  - Có thể cấu hình **fallback embedding cục bộ** bằng `sentence-transformers` (ví dụ model đa ngôn ngữ như `intfloat/multilingual-e5-base`) cho trường hợp: (a) cần chạy offline/không phụ thuộc mạng, (b) muốn tránh chi phí gọi API embedding khi ingest khối lượng lớn bài viết, hoặc (c) API Qwen embedding gặp sự cố tạm thời trong lúc reindex.
  - **Trường hợp chỉ dùng Groq** (không có key Qwen/DeepSeek): embedding cục bộ (`sentence-transformers`) không còn là fallback mà là **lựa chọn bắt buộc duy nhất**, vì Groq không có API embedding. Toàn bộ chi phí compute embedding lúc này chuyển sang chạy trên chính server/container của bạn (CPU hoặc GPU nếu có) thay vì gọi API ngoài.
  - Cần đảm bảo **query lúc trả lời và toàn bộ chunk lúc index dùng cùng một embedding model** (cùng dimension) — nếu đổi giữa Qwen embedding API và model cục bộ, bắt buộc phải **rebuild toàn bộ collection**, không thể trộn lẫn 2 loại vector cùng dimension khác nhau trong 1 collection.
- Chunk được embed và ghi vào một **Qdrant collection** (chạy dạng server — self-host qua Docker hoặc Qdrant Cloud), dữ liệu được **persist thật sự** trên đĩa/volume, không mất khi service restart.
- Mỗi lần khởi động service **không cần rebuild toàn bộ index** — chỉ cần kết nối tới Qdrant collection đã có sẵn. Việc "full ingestion" (đọc toàn bộ bài viết `published` từ DB rồi index) chỉ cần chạy **một lần đầu** khi khởi tạo collection, hoặc chạy thủ công khi cần rebuild hoàn toàn (ví dụ đổi embedding model).
- Mỗi điểm (point) trong Qdrant dùng `point_id` xác định (ví dụ hash từ `article_id` + `chunk_index`), giúp việc upsert/xoá theo từng chunk cụ thể rất hiệu quả mà không đụng tới các point khác.

### 3.5. Đồng Bộ Khi Dữ Liệu Thay Đổi — Cập Nhật Gia Tăng (Incremental), Không Rebuild Toàn Bộ

Vì dữ liệu nằm trong database (không còn là file), việc phát hiện thay đổi được thực hiện theo một (hoặc kết hợp) các cách sau:

1. **SQLAlchemy ORM event hook** (khuyến nghị, real-time nhất): đăng ký listener trên các sự kiện `after_insert`, `after_update`, `after_delete` của model `Article`. Khi có thay đổi, đẩy `article_id` + loại thao tác vào một hàng đợi (in-process queue hoặc Redis queue) để xử lý bất đồng bộ, tránh chặn transaction chính.
2. **Outbox pattern** (khi cần đảm bảo an toàn hơn, nhiều instance): mỗi lần insert/update/delete `article` trong cùng transaction sẽ ghi thêm 1 dòng vào bảng `outbox_events` (`article_id`, `action`, `processed`, `created_at`). Một worker nền định kỳ đọc các event chưa `processed`, thực hiện reindex, rồi đánh dấu đã xử lý.
3. **Polling định kỳ** (đơn giản nhất, dùng khi không cần real-time tuyệt đối): dùng `APScheduler` chạy job mỗi N giây/phút, so sánh `updated_at` (tín hiệu chính) với lần index gần nhất để tìm các bài viết cần reindex; `content_hash` chỉ dùng để xác nhận phụ khi cần loại trừ trường hợp `updated_at` đổi nhưng nội dung thực chất không đổi.

Xử lý theo từng loại thao tác — **tất cả đều là thao tác gia tăng (incremental), chỉ tác động đúng các point liên quan tới `article_id` đó, không cần rebuild toàn bộ collection**:

- **Bài viết mới / cập nhật nội dung** (`updated_at` đổi, xác nhận thêm qua `content_hash` nếu cần): xoá các point cũ có `article_id` tương ứng (`client.delete(collection_name, points_selector=Filter(must=[FieldCondition(key="article_id", match=MatchValue(value=id))]))`), chunk lại nội dung mới, embed lại, `upsert` các point mới vào Qdrant.
- **Bài viết bị xoá, hoặc chuyển từ `published` sang `draft`**: xoá toàn bộ point có `article_id` tương ứng khỏi collection bằng filter theo payload, tương tự trên.
- **Đổi category của bài viết**: vì Qdrant hỗ trợ **cập nhật payload tại chỗ** (`set_payload`) mà không cần tính lại vector, chỉ cần cập nhật `category_id`/`category_name`/`category_slug` trong payload của các point thuộc `article_id` đó — không cần xoá và embed lại.

Cơ chế **debounce** (gộp các thay đổi trong ~1-2 giây) được áp dụng để tránh upsert/xoá liên tục khi một bài viết được lưu nháp nhiều lần liên tiếp trong thời gian ngắn. Việc đồng bộ luôn chạy trong background task/worker riêng, không chặn luồng xử lý request của API hay giao diện quản trị (CMS) khi người dùng lưu bài viết.

---

## 4. Quản Lý Nhiều LLM Qua LangChain

- Mỗi provider (DeepSeek, Groq, Qwen) được khởi tạo thành một `ChatModel` riêng biệt trong LangChain (ví dụ `ChatDeepSeek`, `ChatGroq`, hoặc cấu hình `ChatOpenAI` trỏ base_url tương thích cho Qwen nếu dùng endpoint dạng OpenAI-compatible).
- Xây dựng lớp **Model Router** — áp dụng cho bước **sinh câu trả lời (chat/completion)**, không áp dụng cho embedding (embedding dùng riêng API Qwen embedding, xem mục 3.4):
  - Cấu hình model mặc định.
  - Cho phép người dùng chọn model qua tham số request (`model: "deepseek" | "groq" | "qwen"`).
  - Cơ chế **fallback**: nếu model chính lỗi (timeout, rate limit từ phía provider), tự động chuyển sang model dự phòng theo danh sách ưu tiên (dùng `RunnableWithFallbacks` của LangChain).
- RAG chain dùng chung 1 prompt template cho mọi model, đảm bảo tính nhất quán câu trả lời dù chọn model nào.

---

## 5. Lưu Trữ Hội Thoại (Chat History)

### 5.1. Mô hình dữ liệu

- **Bảng `sessions`**: `session_id`, `user_id`, `created_at`, `title` (tóm tắt tự động).
- **Bảng `messages`**: `message_id`, `session_id`, `role` (`user`/`assistant`), `content`, `sources` (danh sách nguồn dùng để trả lời — mỗi nguồn gồm `article_id`, `title`, `heading_path`, `url` dựng từ `{base_url}/{category_slug}/{article_slug}`, `thumbnail_url`, `description` — đủ để dựng rich preview card khi hiển thị link), `model_used`, `created_at`.

### 5.2. Cách hoạt động

- Mỗi lượt chat: lưu message của người dùng trước, sau khi có phản hồi thì lưu message của assistant kèm metadata nguồn trích dẫn và model đã dùng.
- Lịch sử hội thoại gần nhất (N lượt gần nhất hoặc theo giới hạn token) được đưa vào context của RAG chain để hỗ trợ hỏi đáp nhiều lượt (multi-turn), dùng `RunnableWithMessageHistory` của LangChain kết hợp với lớp lưu trữ tuỳ chỉnh đọc/ghi từ database.
- API cho phép người dùng liệt kê session, xem lại message theo session, xoá session.

---

## 6. Cơ Chế Rate Limit

- Giới hạn theo **user_id** (nếu có xác thực) hoặc **IP** (nếu ẩn danh).
- Áp dụng 2 tầng:
  - **Giới hạn số request** trong cửa sổ thời gian (ví dụ: 20 request/phút) — dùng thuật toán **sliding window** hoặc **token bucket**.
  - **Giới hạn theo số token** sử dụng trong ngày (áp cho từng user) để kiểm soát chi phí gọi LLM API.
- Có thể triển khai bằng:
  - `slowapi` (middleware cho FastAPI, backend lưu counter bằng in-memory hoặc Redis) cho trường hợp đơn giản.
  - Tự viết token-bucket dùng Redis (`INCR` + `EXPIRE`) nếu cần scale nhiều instance.
- Khi vượt giới hạn: trả về HTTP 429 kèm thông tin thời gian còn lại trước khi được phép gọi tiếp (header `Retry-After`).

---

## 7. API Endpoints (Đề Xuất)

| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/chat` | Gửi câu hỏi, nhận câu trả lời RAG (kèm `session_id`, `model` tuỳ chọn) |
| GET | `/sessions` | Danh sách session của người dùng |
| GET | `/sessions/{session_id}/messages` | Lịch sử tin nhắn của 1 session |
| DELETE | `/sessions/{session_id}` | Xoá session |
| POST | `/admin/reindex` | Kích hoạt reindex thủ công toàn bộ knowledge base (toàn bộ bài viết `published`) |
| GET | `/admin/index-status` | Trạng thái vector store (số chunk, thời điểm cập nhật gần nhất) |
| POST/PUT/DELETE | `/admin/categories`, `/admin/articles` | CRUD category/bài viết (nội dung Markdown) — mỗi thao tác tự động kích hoạt đồng bộ vector index tương ứng |

---

## 8. Cấu Trúc Thư Mục Đề Xuất

```
project/
├── .github/
│   └── workflows/
│       └── docker-publish.yml  # CI/CD: build & push image lên Docker Hub
├── app/
│   ├── main.py                # Khởi tạo FastAPI app
│   ├── api/
│   │   ├── chat.py            # Endpoint /chat
│   │   └── sessions.py        # Endpoint quản lý session
│   ├── core/
│   │   ├── config.py          # Đọc & validate biến môi trường (Pydantic Settings) — chi tiết mục 12
│   │   └── rate_limiter.py    # Middleware rate limit
│   ├── rag/
│   │   ├── loader.py          # Loader tuỳ chỉnh: đọc articles (Markdown) từ DB
│   │   ├── chunker.py         # Paragraph chunking
│   │   ├── vectorstore.py     # Khởi tạo & thao tác Qdrant client/collection
│   │   ├── sync.py            # Event hook / outbox worker / polling đồng bộ index
│   │   ├── chain.py           # RAG chain (retriever + prompt + LLM)
│   │   └── source_links.py    # Dựng nguồn tham khảo: url, thumbnail_url, description (rich preview)
│   ├── llm/
│   │   └── router.py          # Model router: DeepSeek/Groq/Qwen + fallback
│   ├── history/
│   │   ├── models.py          # SQLAlchemy models (sessions, messages)
│   │   └── store.py           # Đọc/ghi lịch sử cho LangChain
│   ├── content/
│   │   └── models.py          # SQLAlchemy models (categories, articles)
│   └── db/
│       └── database.py        # Kết nối DB
├── Dockerfile                  # Build image cho app FastAPI
├── docker-compose.yml          # Orchestrate app + Postgres + Qdrant + Redis
├── .dockerignore
├── requirements.txt
└── .env
```

---

## 9. Luồng Xử Lý 1 Request Chat (Tóm Tắt)

1. Client gửi `POST /chat` kèm `session_id`, `message`, `model` (tuỳ chọn).
2. Middleware rate limit kiểm tra quota của user/IP → nếu vượt, trả 429.
3. Lấy lịch sử hội thoại gần nhất từ DB theo `session_id`.
4. Retriever truy vấn Qdrant lấy top-k đoạn văn liên quan.
5. Prompt template kết hợp: lịch sử hội thoại + đoạn văn liên quan + câu hỏi hiện tại.
6. Gửi tới LLM router → gọi model được chọn (có fallback nếu lỗi).
7. Từ metadata các chunk đã dùng (`category_slug`, `article_slug`, `thumbnail_url`, `meta_description`), dựng thông tin nguồn đầy đủ cho từng bài viết: `url` theo công thức `{base_url}/{category_slug}/{article_slug}`, kèm `thumbnail_url` và `description` để phía client render rich preview card (thay vì chỉ hiện link trần).
8. Lưu message người dùng và message trả lời (kèm danh sách nguồn trích dẫn — gồm cả url/thumbnail/description — và model đã dùng) vào DB.
9. Trả kết quả về client, kèm danh sách nguồn tham khảo (`title`, `heading_path`, `url`, `thumbnail_url`, `description`).

---

## 10. Các Điểm Cần Lưu Ý Khi Triển Khai

- Vì Qdrant persistent, service **không cần rebuild toàn bộ index khi restart** — chỉ cần đảm bảo full ingestion chạy đúng **một lần** lúc khởi tạo collection lần đầu (hoặc khi chủ động rebuild, ví dụ đổi embedding model/kích thước vector).
- Chạy nhiều worker/instance API là an toàn vì tất cả cùng trỏ tới **một Qdrant server/collection tập trung** — không còn vấn đề "mỗi instance có index riêng" như với Chroma in-memory. Tuy nhiên vẫn cần lưu ý: nếu dùng ORM event hook, event chỉ bắn ra trên instance thực hiện ghi DB, nên nên đẩy qua queue chung (Redis/outbox) để các worker khác cũng nắm được thay đổi, tránh phụ thuộc vào đúng 1 instance xử lý sync.
- Cần xử lý đồng thời (concurrency) khi nhiều bài viết được lưu/cập nhật cùng lúc — dùng lock hoặc queue cho tác vụ reindex, tránh 2 worker cùng ghi/xoá chồng lên nhau trên cùng `article_id`.
- Cân nhắc giới hạn kích thước nội dung Markdown của mỗi bài viết để tránh reindex quá nặng.
- Log lại toàn bộ lần reindex (`article_id` nào, category nào, thời gian, số chunk thay đổi) để dễ debug và audit.
- Khi xoá "mềm" (soft delete, chỉ đổi `status`) thay vì xoá hẳn record, vẫn cần coi đó là sự kiện xoá khỏi vector index nếu bài viết không còn ở trạng thái `published`.
- Vì chỉ Qwen có API embedding (DeepSeek/Groq không hỗ trợ), cần theo dõi rate limit/quota riêng của API embedding Qwen — tách biệt với quota của API chat Qwen, để tránh việc ingest hàng loạt bài viết làm ảnh hưởng tới khả năng phục vụ chat.
- Nếu sau này đổi embedding model (ví dụ chuyển từ Qwen embedding API sang model cục bộ, hoặc đổi version embedding có dimension khác), bắt buộc **rebuild toàn bộ Qdrant collection** — nên có sẵn script/CLI command riêng cho việc này (đã liệt kê ở `/admin/reindex`), và cân nhắc dựng collection mới song song rồi chuyển traffic (blue-green) thay vì xoá collection cũ ngay lập tức.

---

## 12. File Cấu Hình (`.env` / `config.py`)

`app/core/config.py` dùng **Pydantic Settings** (`BaseSettings`) để đọc và validate toàn bộ biến môi trường ngay lúc khởi động — app sẽ báo lỗi rõ ràng và **fail fast** nếu thiếu biến bắt buộc, thay vì lỗi mơ hồ lúc runtime khi có request đầu tiên.

### 12.1. Nhóm LLM Providers

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `GROQ_API_KEY` | Có (nếu dùng Groq) | API key gọi model chat qua Groq |
| `GROQ_MODEL` | Không (có default) | Tên model chat mặc định của Groq, ví dụ `llama-3.3-70b-versatile` |
| `DEEPSEEK_API_KEY` | Tuỳ chọn | API key DeepSeek — bỏ trống nếu không dùng provider này |
| `DEEPSEEK_MODEL` | Không | Tên model chat DeepSeek mặc định |
| `QWEN_API_KEY` | Tuỳ chọn | API key Qwen (DashScope/Model Studio) — dùng cho cả chat lẫn embedding nếu có |
| `QWEN_MODEL` | Không | Tên model chat Qwen mặc định |
| `QWEN_BASE_URL` | Tuỳ chọn | Base URL endpoint tương thích OpenAI của DashScope, nếu khác default |
| `DEFAULT_CHAT_PROVIDER` | Có | Provider dùng làm mặc định khi request không chỉ định `model` — ví dụ `groq` nếu chỉ có Groq |
| `CHAT_FALLBACK_ORDER` | Không | Thứ tự fallback khi model chính lỗi, dạng danh sách phân tách bởi dấu phẩy, ví dụ `groq` (chỉ 1 phần tử nếu chỉ có Groq) hoặc `qwen,deepseek,groq` |

### 12.2. Nhóm Embedding

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `EMBEDDING_PROVIDER` | Có | `qwen_api` hoặc `local` — quyết định dùng API Qwen embedding hay `sentence-transformers` cục bộ. **Nếu chỉ có Groq (không có `QWEN_API_KEY`), bắt buộc phải set giá trị này là `local`** |
| `LOCAL_EMBEDDING_MODEL` | Có nếu `EMBEDDING_PROVIDER=local` | Tên model `sentence-transformers`, ví dụ `intfloat/multilingual-e5-base` |
| `EMBEDDING_DIMENSION` | Có | Dimension vector — phải khớp với model đang dùng, dùng để tạo Qdrant collection đúng kích thước |

### 12.3. Nhóm Database & Vector Store

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `DATABASE_URL` | Có | Connection string PostgreSQL/MySQL cho `categories`, `articles`, `sessions`, `messages` |
| `QDRANT_URL` | Có | Địa chỉ Qdrant server, ví dụ `http://qdrant:6333` trong Docker Compose |
| `QDRANT_API_KEY` | Tuỳ chọn | Chỉ cần nếu Qdrant bật xác thực (ví dụ Qdrant Cloud) |
| `QDRANT_COLLECTION_NAME` | Có | Tên collection lưu vector knowledge base |
| `REDIS_URL` | Có | Dùng cho rate limit và/hoặc hàng đợi đồng bộ index, ví dụ `redis://redis:6379/0` |

### 12.4. Nhóm Rate Limit

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | Không (có default) | Số request tối đa mỗi user/IP trong 1 phút |
| `RATE_LIMIT_TOKENS_PER_DAY` | Không | Giới hạn số token gọi LLM mỗi user mỗi ngày |

### 12.5. Nhóm Ứng Dụng Chung

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `SITE_BASE_URL` | Có | Dùng để dựng link chi tiết bài viết: `{SITE_BASE_URL}/{category_slug}/{article_slug}` |
| `ENV` | Không (default `production`) | `development`/`staging`/`production` — bật/tắt log chi tiết, docs Swagger, v.v. |
| `LOG_LEVEL` | Không (default `INFO`) | Mức log cho app và sync worker |
| `SYNC_DEBOUNCE_SECONDS` | Không (có default) | Thời gian debounce trước khi reindex một bài viết vừa thay đổi (mục 3.5) |

### 12.6. Ví Dụ File `.env.example`

```env
# --- LLM Providers ---
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=
QWEN_API_KEY=
QWEN_MODEL=
QWEN_BASE_URL=
DEFAULT_CHAT_PROVIDER=groq
CHAT_FALLBACK_ORDER=groq

# --- Embedding ---
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIMENSION=768

# --- Database & Vector Store ---
DATABASE_URL=postgresql+psycopg2://user:password@postgres:5432/ragchatbot
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=knowledge_base
REDIS_URL=redis://redis:6379/0

# --- Rate Limit ---
RATE_LIMIT_PER_MINUTE=20
RATE_LIMIT_TOKENS_PER_DAY=50000

# --- App ---
SITE_BASE_URL=https://example.com
ENV=production
LOG_LEVEL=INFO
SYNC_DEBOUNCE_SECONDS=2
```

Ví dụ trên là cấu hình cho **kịch bản chỉ có Groq**: `DEEPSEEK_API_KEY`/`QWEN_API_KEY` để trống, `DEFAULT_CHAT_PROVIDER` và `CHAT_FALLBACK_ORDER` chỉ có `groq`, và `EMBEDDING_PROVIDER=local` bắt buộc vì không có provider nào khác cấp API embedding.

Chỉ commit `.env.example` (không chứa giá trị thật) vào git; file `.env` thật chứa secret nằm trong `.gitignore` và `.dockerignore`.

---

## 13. CI/CD — GitHub Actions Đẩy Image Lên Docker Hub

### 13.1. Cơ Chế Hoạt Động

File `.github/workflows/docker-publish.yml` định nghĩa 2 job:

- **`build-only`**: chạy khi có pull request nhắm vào `main` — chỉ build Docker image để xác nhận `Dockerfile` không lỗi, **không push** lên Docker Hub. Giúp phát hiện lỗi build sớm ngay từ PR review, trước khi merge.
- **`build-and-push`**: chạy khi push trực tiếp vào `main` hoặc khi push git tag dạng `vX.Y.Z` — build image rồi đẩy lên Docker Hub với tag tương ứng:
  - Push vào `main` → gắn tag `latest` + tag ngắn theo commit SHA (dễ trace lại đúng bản build).
  - Push tag `v1.2.3` → gắn tag semver đầy đủ (`1.2.3`) và tag `major.minor` (`1.2`).
- Dùng `docker/build-push-action` kết hợp cache qua GitHub Actions cache (`type=gha`) để các lần build sau nhanh hơn nhờ tái sử dụng layer.
- Build đa kiến trúc (`linux/amd64,linux/arm64`) qua `docker/setup-qemu-action`, phù hợp nếu bạn deploy trên cả server x86 lẫn ARM (ví dụ AWS Graviton).

Nội dung đầy đủ file `.github/workflows/docker-publish.yml`:

```yaml
name: Build and Push Docker Image

# Kích hoạt khi:
# - push lên nhánh main (build + push tag "latest")
# - push tag dạng semver v1.2.3 (build + push tag version tương ứng)
# - pull request nhắm vào main (chỉ build để validate, KHÔNG push)
# - có thể chạy tay qua tab Actions (workflow_dispatch)
on:
  push:
    branches: ["main"]
    tags: ["v*.*.*"]
  pull_request:
    branches: ["main"]
  workflow_dispatch:

env:
  # Tên image trên Docker Hub, dạng <docker_hub_username>/<repo_name>
  # DOCKERHUB_USERNAME lấy từ GitHub Secrets, DOCKERHUB_REPO có thể set cứng hoặc cũng để trong Secrets/Variables
  IMAGE_NAME: ${{ secrets.DOCKERHUB_USERNAME }}/${{ vars.DOCKERHUB_REPO || 'rag-chatbot' }}

jobs:
  # Job build-only: chạy cho pull request, chỉ để đảm bảo Dockerfile build được, KHÔNG đẩy lên Docker Hub
  build-only:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build image (no push)
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: false
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Job build-and-push: chạy khi push vào main hoặc push tag semver, đẩy image lên Docker Hub
  build-and-push:
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up QEMU (hỗ trợ build đa kiến trúc)
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      # Tự động sinh tag cho image dựa trên sự kiện trigger:
      # - nhánh main -> tag "latest"
      # - commit sha  -> tag ngắn theo sha (dễ trace lại đúng bản build)
      # - git tag vX.Y.Z -> tag semver đầy đủ + tag major.minor
      - name: Extract Docker metadata (tags, labels)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,prefix=,format=short
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Build and push image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Update Docker Hub description (tuỳ chọn)
        if: github.ref == 'refs/heads/main'
        uses: peter-evans/dockerhub-description@v4
        continue-on-error: true
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
          repository: ${{ env.IMAGE_NAME }}
          readme-filepath: ./README.md
```

### 13.2. Secrets Cần Cấu Hình Trên GitHub Repository

Vào **Settings → Secrets and variables → Actions** của repo GitHub, khai báo:

| Tên Secret/Variable | Loại | Bắt buộc | Mô tả |
|---|---|---|---|
| `DOCKERHUB_USERNAME` | Secret | Có | Username tài khoản Docker Hub của bạn |
| `DOCKERHUB_TOKEN` | Secret | Có | **Access Token** tạo từ Docker Hub (Account Settings → Security → New Access Token) — **không dùng mật khẩu tài khoản trực tiếp** vì kém an toàn hơn và không thu hồi được riêng lẻ |
| `DOCKERHUB_REPO` | Variable (không phải Secret, vì không nhạy cảm) | Không (có default `rag-chatbot` trong workflow) | Tên repository trên Docker Hub, ví dụ `rag-chatbot` — image cuối sẽ có dạng `{DOCKERHUB_USERNAME}/{DOCKERHUB_REPO}` |

Lưu ý phân biệt: GitHub Actions có 2 khái niệm riêng — **Secrets** (giá trị mã hoá, không hiện lại được sau khi lưu, dùng cho token/password) và **Variables** (giá trị plain-text, xem lại được, dùng cho cấu hình không nhạy cảm như tên repo). `DOCKERHUB_REPO` nên để dạng Variable vì không phải thông tin nhạy cảm.

### 13.3. Các Bước Thiết Lập Trên Docker Hub

1. Đăng nhập Docker Hub → **Account Settings → Security → New Access Token**, đặt quyền **Read & Write**, copy token này để dán vào `DOCKERHUB_TOKEN` trên GitHub (token chỉ hiện 1 lần lúc tạo).
2. Tạo trước repository trên Docker Hub (public hoặc private tuỳ nhu cầu) với tên khớp với `DOCKERHUB_REPO`, hoặc để workflow tự tạo repository mới khi push lần đầu (Docker Hub cho phép tạo repo qua lần push đầu tiên nếu tài khoản có quyền).

### 13.4. Quy Trình Release Đề Xuất

1. Merge code vào `main` → tự động build + push tag `latest` (dùng cho môi trường staging/luôn cập nhật mới nhất).
2. Khi sẵn sàng release chính thức, tạo git tag theo semver và push tag đó:

```bash
git tag v1.0.0
git push origin v1.0.0
```

3. Workflow tự động build + push image với tag `1.0.0` và `1.0` lên Docker Hub — dùng tag cố định này cho `docker-compose.yml` ở môi trường production, thay vì `latest`, để đảm bảo mỗi lần deploy là một phiên bản xác định, có thể rollback dễ dàng bằng cách trỏ lại tag cũ.

---

## 11. Đóng Gói Docker

### 11.1. Dockerfile Cho App (FastAPI)

- Dùng **multi-stage build**: stage 1 (`builder`) cài dependency vào virtualenv/wheel riêng, stage 2 (`runtime`) chỉ copy các package đã build sang, giúp image cuối gọn hơn, không mang theo cache pip hay công cụ build (gcc, ...).
- Base image: `python:3.11-slim` cho cả 2 stage.
- Chạy app bằng `uvicorn` (hoặc `gunicorn` với `uvicorn.workers.UvicornWorker` cho production, cho phép nhiều worker process).
- Không copy file `.env` vào image — biến môi trường được truyền vào lúc chạy container qua `docker-compose.yml`/`--env-file`, tránh lộ secret (API key DeepSeek/Groq/Qwen) trong image.
- Chạy app bằng user không phải root trong container (tạo user riêng, `USER appuser`) để tăng bảo mật.

### 11.2. Docker Compose — Các Service

| Service | Vai trò |
|---|---|
| `app` | FastAPI app (RAG chain, chat API, rate limit, sync worker) — build từ `Dockerfile` |
| `postgres` | Lưu `categories`, `articles`, `sessions`, `messages` — dùng image chính thức `postgres`, mount volume để persist dữ liệu |
| `qdrant` | Vector database — dùng image chính thức `qdrant/qdrant`, mount volume để persist collection |
| `redis` | Backend cho rate limit (token bucket) và/hoặc hàng đợi cho cơ chế đồng bộ index (outbox worker / event queue) |

- `app` khai báo `depends_on` với `postgres`, `qdrant`, `redis`, kết hợp `healthcheck` cho từng service phụ thuộc để `app` chỉ khởi động sau khi các service nền tảng đã sẵn sàng (tránh lỗi kết nối lúc container mới up).
- Volume riêng cho `postgres` (`pg_data`) và `qdrant` (`qdrant_data`) để dữ liệu không mất khi `docker compose down` (không dùng `down -v` nếu muốn giữ dữ liệu).
- Biến môi trường (API key LLM, connection string DB/Qdrant/Redis, `SITE_BASE_URL`) được truyền qua file `.env` (không commit vào git, chỉ commit `.env.example` làm mẫu).
- Có thể tách thêm 1 service `sync-worker` riêng (cùng image với `app` nhưng chạy entrypoint khác) nếu muốn tách tiến trình đồng bộ index (outbox worker / APScheduler) ra khỏi tiến trình phục vụ API, giúp scale độc lập và tránh một tác vụ nặng ảnh hưởng tới độ trễ trả lời chat.

### 11.3. Một Số Lưu Ý Khi Đóng Gói

- File `.dockerignore` loại trừ `__pycache__`, `.venv`, `.git`, `.env`, file test, giúp build nhanh hơn và image nhỏ gọn hơn.
- Đặt `HEALTHCHECK` cho service `app` (gọi tới endpoint dạng `/healthz`) để Compose/orchestrator (Docker Swarm, Kubernetes sau này) biết khi nào container thật sự sẵn sàng nhận traffic.
- Với embedding model chạy cục bộ (`sentence-transformers`), cân nhắc tải sẵn model vào image lúc build (thay vì tải lúc runtime) để tránh cold-start chậm và tránh phụ thuộc mạng khi container khởi động.
- Migration schema database (ví dụ Alembic) nên chạy như một bước riêng (init container hoặc lệnh `docker compose run app alembic upgrade head`) trước khi `app` chính thức nhận traffic, không tự động chạy ngầm trong quá trình khởi động app.
- Với môi trường production nhiều node, nên chuyển `qdrant`/`postgres`/`redis` sang dịch vụ managed (Qdrant Cloud, RDS, ElastiCache/Redis Cloud) thay vì tự chạy container để giảm gánh nặng vận hành, đặc biệt là backup và high-availability.
