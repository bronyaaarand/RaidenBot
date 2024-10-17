### Tổng Quan Dự Án

Dự án này bao gồm hai thành phần chính:

1. **Ứng Dụng Di Động**: Ứng dụng di động đóng vai trò là giao diện để tương tác với cả Zalo và Dify AI. Nó cho phép người dùng sử dụng các tính năng do nền tảng Zalo cung cấp và dịch vụ của Dify AI một cách liền mạch.

2. **API**: API backend chịu trách nhiệm đảm bảo sự tương tác mượt mà giữa ứng dụng di động và các dữ liệu cần thiết. Nó hoạt động như một cầu nối để hỗ trợ trao đổi và xử lý dữ liệu cần thiết cho các chức năng của ứng dụng.

Dự án tích hợp các công nghệ trên để cung cấp cho người dùng trải nghiệm di động tương tác và nhạy bén.

#### Cấu Trúc

- `mobile_app/`: Chứa mã nguồn của ứng dụng di động.
- `api_service/`: Bao gồm API chịu trách nhiệm quản lý các tương tác dữ liệu.

#### Các Tính Năng

- Tích hợp Zalo: Tương tác với Zalo để cung cấp các dịch vụ đặc trưng của nền tảng.
- Tích hợp Dify AI: Cho phép ứng dụng sử dụng khả năng của Dify AI để phản hồi thông minh.

### Cấu trúc Ứng dụng Di động và Luồng Dữ liệu

Ứng dụng di động này được phát triển bằng **React Native**, với mục đích hỗ trợ nhân viên chăm sóc khách hàng trong việc chuyển tiếp thông tin hai chiều giữa **Ai - RaidenBot** và **User** từ **Zalo**. Ứng dụng cung cấp một giao diện trực quan, cho phép nhân viên dễ dàng tương tác với người dùng và bot, từ đó nâng cao hiệu quả trong việc đáp ứng các yêu cầu báo cáo thống kê một cách tự động.


#### Mục tiêu Ứng dụng

- **Tương tác hai chiều**: Cung cấp khả năng giao tiếp trực tiếp giữa nhân viên chăm sóc khách hàng và người dùng thông qua bot AI.
- **Hỗ trợ báo cáo**: Tự động hóa quy trình báo cáo và thống kê thông tin liên quan đến người dùng, giúp tiết kiệm thời gian và công sức cho nhân viên.
- **Tích hợp API Zalo**: Sử dụng API từ Zalo để lấy thông tin người dùng, giúp quản lý và tương tác hiệu quả hơn.
- **Giao diện thân thiện**: Thiết kế giao diện dễ sử dụng, giúp nhân viên nhanh chóng làm quen và sử dụng ứng dụng.
---

#### Cấu trúc Thư mục

```bash
app/
│
├── (tabs)/
│   ├── bot.tsx       
│   ├── person.tsx    
│   └── button.tsx    
│
├── assets/           
│
├── index.tsx        
│
└── App.tsx          

```

#### Tổng quan về các Màn hình của Ứng dụng

#### 1. Màn hình Chính (`index.tsx`)
- Hiển thị danh sách người dùng được lấy từ API Zalo.
- Cho phép điều hướng đến màn hình `button` khi người dùng được nhấp.
- Lấy `user_id`, `user_name` và `user_avatar` cho mỗi người dùng.
- Gửi dữ liệu đến màn hình `button` bằng phương thức `router.push()` với các tham số.

#### 2. Màn tùy chọn thao tác (`button.tsx`)
- Nhận các tham số (`user_id`, `user_name`, `user_avatar`) từ Màn hình Chính.
- Hiển thị thông tin của người dùng (tên, avatar) và cho phép tùy chọn trò chuyện trực tiếp với người dùng hoặc kiểm tra phản hồi của RaidenBot.
- Điều hướng đến màn hình `person.tsx` với `user_id` đã chọn.

#### 3. Màn User (`person.tsx`)
- Lấy dữ liệu trò chuyện cho `user_id` đã cho từ API Zalo.
- Hiển thị lịch sử trò chuyện cho người dùng được chỉ định.

#### 4. Màn Bot (`bot.tsx`)
- Hiển thị giao diện trò chuyện với RaidenBot.

```plaintext
+-----------------------+
|     HomeScreen        |
|     (index.tsx)      |
+-----------+-----------+
            |
            | Chọn Người Dùng
            v
+-----------------------+
|     ButtonScreen      |
|     (button.tsx)     |
+-----------+-----------+
            |
            | Chuyển Giao Diện
            v
+-----------------------+
|     PersonScreen      |
|     (person.tsx)     |
+-----------+-----------+
            |
            | Lấy Dữ Liệu Trò Chuyện
            v
+-----------------------+
|       API Zalo       |
+-----------------------+
            ^
            |
            | Chọn Người Dùng
            |
+-----------------------+
|      BotScreen        |
|      (bot.tsx)       |
+-----------------------+
```

### Cấu Trúc API

Backend của dự án RaidenBot được tổ chức dưới 1 API xây dựng bằng Flask, mục đích trung gian cho sự tương tác giữa Ứng dụng Di động (Mobile App) và API từ Dify, đồng thời quản lý dữ liệu lưu trữ trong MongoDB.

#### Các Thành Phần Chính

- **Thư viện sử dụng**:
  - `Flask`: Khung ứng dụng web.
  - `pymongo`: Kết nối và làm việc với MongoDB.
  - `requests`: Gửi yêu cầu HTTP đến API Dify.

- **Kết nối đến MongoDB**:
  - Kết nối với cơ sở dữ liệu `raiden` và hai collection: `const` (chứa thông tin mã truy cập) và `dify_response` (lưu trữ các phản hồi từ Dify AI).

- **Nhãn intents**:
  - Định nghĩa các nhãn cho các loại yêu cầu như `about`, `statistic`, và `contact`.

#### Các Endpoint

1. **`/access-token`**: 
   - **Phương thức**: `GET`
   - **Mô tả**: Trả về mã truy cập từ cơ sở dữ liệu để Ứng dụng Di động có thể sử dụng.

2. **`/dify-message`**: 
   - **Phương thức**: `POST`
   - **Mô tả**: Nhận tin nhắn từ người dùng, gửi đến API Dify để nhận phản hồi, và lưu trữ phản hồi trong MongoDB.

3. **`/dify-history`**:
   - **Phương thức**: `GET`
   - **Mô tả**: Trả về danh sách các tin nhắn đã gửi và phản hồi từ Dify AI, giúp Ứng dụng Di động theo dõi lịch sử tương tác.

### Cấu Hình Các Dịch Vụ Từ Bên Ngoài

#### 1. Dify AI

Để cấu hình Dify AI, làm theo các bước sau:

- Truy cập vào https://dify.ai và đăng nhập.
- Tạo một ứng dụng chatflow mới.
- Chọn tùy chọn nhập DSL và nhập file `Raiden.yaml`.
- Tùy chỉnh các prompt trên trang web theo ý muốn.

#### 2. MongoDB

MongoDB cần đảm bảo có ba collection sau:

#### 2.1. `dify_response`
- Cập nhật liên tục từ Ứng dụng Di động và API của RaidenBot.
- Lưu lại tin nhắn phản hồi từ RaidenBot.

#### Cấu trúc trường:
"message_id"; "customer_request"; "agent_response"

#### 2.2. `reports`
- Được cập nhật mỗi lần có báo cáo mới từ Looker Studio.
- Cập nhật trực tiếp vào bộ dữ liệu tri thức của Dify AI.
#### Cấu trúc trường:
"factor"; "type"; "day"; "month"; "year"; "from_date"; "to_date"; "agent"; "link"; "slug"

#### 2.3. `const`
- Lưu trữ thông tin cấu hình như mã truy cập.


#### Cấu trúc trường:
"name"; "value"; "expires_in"

### Deploy

#### 1. Deploy API Lên Render

#### Bước 1: Đăng Nhập Render Bằng GitHub
- Truy cập https://render.com và đăng nhập bằng tài khoản GitHub.

#### Bước 2: Tạo Project Mới
- Nhấn **New** và chọn **Web Service**.

#### Bước 3: Chọn Repository Chứa Mã Nguồn API
- Chọn repository chứa mã nguồn API của RaidenBot.

#### Bước 4: Cấu Hình Câu Lệnh Khởi Tạo
- Nhập câu lệnh khởi tạo: `gunicorn main:app`.

#### Bước 5: Triển Khai Web Service
- Nhấn **Deploy Web Service**. 
- Sau khi hoàn thành, Render sẽ cung cấp URL API để sử dụng: 
  - **URL hiện tại**: [https://renderbaseapi.onrender.com](https://renderbaseapi.onrender.com).

#### 2. Triển Khai Mobile App Lên Expo (Thêm Sau)
#### Bước 1: Chạy Lệnh Build
- CD vào thư mục chứa source mobile `RaidenBot`, sau đó chạy lệnh:
```bash
eas build -p android --profile preview
```
#### Bước 2: Theo Dõi Quá Trình Build
- Terminal sẽ hiển thị link build log. Truy cập vào link để đăng nhập vào trang Expo và theo dõi quá trình build APK file.
#### Bước 3: Tải APK File
- Tải APK file về sau khi quá trình build hoàn tất.

<a href="https://github.com/bronyaaarand/RaidenBot/releases/download/apk/raiden.apk">
  <img src="https://raw.githubusercontent.com/octalops/MyDis/main/button-apk.png" alt="Download APK" width="150">
</a>

