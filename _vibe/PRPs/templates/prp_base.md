## 目的
針對 AI agent 優化的模板，用於實作具有充分上下文和自我驗證能力的功能，通過迭代優化達到可運行的程式碼。

## 核心原則
1. **Context 至上**：包含所有必要的文件、範例和注意事項
2. **驗證循環**：提供 AI 可以執行和修復的可執行測試/檢查
3. **資訊密集**：使用程式碼庫中的關鍵字和模式
4. **漸進成功**：從簡單開始，驗證，然後增強
5. **全域規則**：確保遵循 CLAUDE.md 中的所有規則

---

## 目標
[需要建構什麼 - 具體說明最終狀態和需求]

## 為什麼
- [商業價值和使用者影響]
- [與現有功能的整合]
- [解決什麼問題以及為誰解決]

## 什麼
[使用者可見的行為和技術需求]

### 成功標準
- [ ] [具體的可衡量結果]

## 所有需要的上下文

### 文件與參考資料 (列出實作功能所需的所有上下文)
```yaml
# 必讀 - 將這些包含在你的上下文視窗中
- url: [官方 API 文件 URL]
  why: [你需要的特定章節/方法]
  
- file: [path/to/example.py]
  why: [要遵循的模式，要避免的陷阱]
  
- doc: [函式庫文件 URL] 
  section: [關於常見陷阱的特定章節]
  critical: [防止常見錯誤的關鍵見解]

- docfile: [PRPs/ai_docs/file.md]
  why: [使用者貼到專案中的文件]

```

### 當前程式碼庫結構 (在專案根目錄執行 `tree` 來獲得程式碼庫概覽)
```bash

```

### 期望的程式碼庫結構，包含要新增的檔案及檔案職責
```bash

```

### 我們程式碼庫的已知陷阱與函式庫特殊性
```python
# 關鍵：[函式庫名稱] 需要 [特定設定]
# 範例：FastAPI 需要 async 函數來處理 endpoints
# 範例：這個 ORM 不支援超過 1000 條記錄的批次插入
# 範例：我們使用 pydantic v2 並且  
```

## 實作藍圖

### 資料模型和結構

建立核心資料模型，確保型別安全性和一致性。
```python
範例: 
 - orm 模型
 - pydantic 模型
 - pydantic schemas
 - pydantic validators

```

### 按完成順序列出完成此 PRP 需要完成的任務清單

```yaml
任務 1:
修改 src/existing_module.py:
  - 尋找模式: "class OldImplementation"
  - 在包含 "def __init__" 的行後插入
  - 保留現有方法簽名

建立 src/new_feature.py:
  - 參考模式來自: src/similar_feature.py
  - 修改類別名稱和核心邏輯
  - 保持錯誤處理模式完全相同

...(...)

任務 N:
...

```

### 每個任務的偽程式碼 (根據需要新增到每個任務)
```python

# 任務 1
# 包含關鍵細節的偽程式碼，不要寫完整程式碼
async def new_feature(param: str) -> Result:
    # 模式：總是先驗證輸入 (參見 src/validators.py)
    validated = validate_input(param)  # 拋出 ValidationError
    
    # 陷阱：這個函式庫需要連接池
    async with get_connection() as conn:  # 參見 src/db/pool.py
        # 模式：使用現有的重試裝飾器
        @retry(attempts=3, backoff=exponential)
        async def _inner():
            # 關鍵：API 如果 >10 req/sec 會返回 429
            await rate_limiter.acquire()
            return await external_api.call(validated)
        
        result = await _inner()
    
    # 模式：標準化回應格式
    return format_response(result)  # 參見 src/utils/responses.py
```

### 整合點
```yaml
DATABASE:
  - migration: "在 users 表中新增欄位 'feature_enabled'"
  - index: "CREATE INDEX idx_feature_lookup ON users(feature_id)"
  
CONFIG:
  - 新增到: config/settings.py
  - 模式: "FEATURE_TIMEOUT = int(os.getenv('FEATURE_TIMEOUT', '30'))"
  
ROUTES:
  - 新增到: src/api/routes.py  
  - 模式: "router.include_router(feature_router, prefix='/feature')"
```

## 驗證循環

### 級別 1：語法與風格
```bash
# 首先執行這些 - 在繼續之前修復任何錯誤
ruff check src/new_feature.py --fix  # 自動修復可能的問題
mypy src/new_feature.py              # 型別檢查

# 預期：沒有錯誤。如果有錯誤，讀取錯誤並修復。
```

### 級別 2：單元測試，每個新功能/檔案/函數使用現有測試模式
```python
# 建立 test_new_feature.py 包含這些測試案例：
def test_happy_path():
    """基本功能正常運作"""
    result = new_feature("valid_input")
    assert result.status == "success"

def test_validation_error():
    """無效輸入拋出 ValidationError"""
    with pytest.raises(ValidationError):
        new_feature("")

def test_external_api_timeout():
    """優雅地處理超時"""
    with mock.patch('external_api.call', side_effect=TimeoutError):
        result = new_feature("valid")
        assert result.status == "error"
        assert "timeout" in result.message
```

```bash
# 執行並迭代直到通過：
uv run pytest test_new_feature.py -v
# 如果失敗：讀取錯誤，理解根本原因，修復程式碼，重新執行 (不要透過 mock 來通過)
```

### 級別 3：元件連接和系統整合測試

#### 3.1 基礎設施連接測試
```bash
# 確保基礎服務運行
docker-compose up -d  # 或啟動 Redis, RabbitMQ 等

# 驗證基礎設施連接
python -c "
import redis
import pika
import psycopg2  # 或其他數據庫
print('Testing Redis...') 
redis.Redis(host='localhost').ping()
print('Testing RabbitMQ...')
pika.BlockingConnection(pika.ConnectionParameters('localhost')).close()
print('All infrastructure services connected!')
"

# 預期：所有服務連接成功，無異常
```

#### 3.2 服務間通信測試
```python
# 建立 tests/integration/test_service_connectivity.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

class TestServiceConnectivity:
    
    @pytest.mark.asyncio
    async def test_service_a_to_service_b_connection(self):
        """測試服務 A 能否成功調用服務 B"""
        from src.services.service_a import ServiceAClient
        from src.services.service_b import ServiceBServer
        
        # 啟動服務 B 的測試實例
        server_b = ServiceBServer()
        await server_b.start_test_server(port=50001)
        
        try:
            # 測試服務 A 連接服務 B
            client_a = ServiceAClient("localhost:50001")
            response = await client_a.call_service_b("test_data")
            
            assert response.success is True
            assert response.data is not None
            
        finally:
            await server_b.stop()
    
    @pytest.mark.asyncio  
    async def test_message_queue_flow(self):
        """測試完整的消息隊列流程"""
        from src.common.rabbitmq_client import RabbitMQClient
        
        client = RabbitMQClient()
        await client.connect()
        
        # 測試發布-訂閱流程
        test_message = {"test": "data", "timestamp": 1234567890}
        
        # 設置消費者接收消息
        received_messages = []
        async def message_handler(message):
            received_messages.append(message)
        
        await client.subscribe("test_queue", message_handler)
        
        # 發布消息
        await client.publish("test_queue", test_message)
        
        # 等待消息處理
        await asyncio.sleep(0.5)
        
        assert len(received_messages) == 1
        assert received_messages[0] == test_message
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_database_to_cache_sync(self):
        """測試數據庫和緩存的同步"""
        from src.common.database import DatabaseClient
        from src.common.redis_client import RedisClient
        
        db = DatabaseClient()
        cache = RedisClient()
        
        # 在數據庫中創建測試數據
        test_data = {"id": "test_001", "value": "test_value"}
        await db.insert("test_table", test_data)
        
        # 驗證緩存更新
        cached_data = await cache.get("test_table:test_001")
        assert cached_data == test_data
        
        # 測試緩存失效
        await db.update("test_table", "test_001", {"value": "updated_value"})
        updated_cached = await cache.get("test_table:test_001") 
        assert updated_cached["value"] == "updated_value"
    
    @pytest.mark.asyncio
    async def test_auth_service_integration(self):
        """測試認證服務整合"""
        from src.services.auth import AuthService
        from src.services.user import UserService
        
        auth_service = AuthService()
        user_service = UserService()
        
        # 測試用戶註冊流程
        user_data = {
            "username": "test_user",
            "email": "test@example.com", 
            "password": "test_password"
        }
        
        # 註冊用戶
        user_id = await user_service.register(user_data)
        assert user_id is not None
        
        # 測試認證
        token = await auth_service.authenticate(
            user_data["username"], 
            user_data["password"]
        )
        assert token is not None
        
        # 驗證 token 有效性
        user_info = await auth_service.verify_token(token)
        assert user_info["username"] == user_data["username"]
```

#### 3.3 端到端工作流測試
```python
# 建立 tests/integration/test_end_to_end_workflows.py
class TestEndToEndWorkflows:
    
    @pytest.mark.asyncio
    async def test_complete_user_journey(self):
        """測試完整的用戶旅程"""
        # 步驟 1: 用戶註冊
        registration_data = {
            "username": "journey_user",
            "email": "journey@example.com"
        }
        
        response = await self.make_request(
            "POST", "/api/register", registration_data
        )
        assert response.status_code == 201
        user_id = response.json()["user_id"]
        
        # 步驟 2: 用戶登入
        login_data = {"username": "journey_user", "password": "password"}
        response = await self.make_request("POST", "/api/login", login_data)
        assert response.status_code == 200
        token = response.json()["token"]
        
        # 步驟 3: 執行主要功能
        headers = {"Authorization": f"Bearer {token}"}
        response = await self.make_request(
            "POST", "/api/main_feature", 
            {"data": "test"}, 
            headers=headers
        )
        assert response.status_code == 200
        
        # 步驟 4: 驗證副作用
        # 檢查數據庫狀態
        user = await self.db_client.get_user(user_id)
        assert user.last_activity is not None
        
        # 檢查緩存狀態
        cache_key = f"user_activity:{user_id}"
        cached_activity = await self.redis_client.get(cache_key)
        assert cached_activity is not None
    
    @pytest.mark.asyncio
    async def test_error_propagation_across_services(self):
        """測試錯誤在服務間的正確傳播"""
        # 模擬下游服務錯誤
        with patch('src.services.downstream.DownstreamService.process') as mock:
            mock.side_effect = Exception("Downstream service error")
            
            # 調用上游服務
            response = await self.make_request(
                "POST", "/api/upstream_endpoint", {"data": "test"}
            )
            
            # 驗證錯誤正確傳播和處理
            assert response.status_code == 500
            assert "Downstream service error" in response.json()["message"]
            
            # 驗證錯誤日誌
            assert "Downstream service error" in self.get_logs()
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_services(self):
        """測試跨服務的數據一致性"""
        # 在服務 A 創建數據
        data_a = {"id": "consistency_test", "value": "initial"}
        response = await self.make_request("POST", "/api/service_a/data", data_a)
        assert response.status_code == 201
        
        # 驗證服務 B 能讀取到同步的數據
        await asyncio.sleep(1)  # 等待異步同步
        response = await self.make_request("GET", "/api/service_b/data/consistency_test")
        assert response.status_code == 200
        assert response.json()["value"] == "initial"
        
        # 在服務 A 更新數據
        update_data = {"value": "updated"}
        response = await self.make_request(
            "PUT", "/api/service_a/data/consistency_test", update_data
        )
        assert response.status_code == 200
        
        # 驗證服務 B 讀取到更新的數據
        await asyncio.sleep(1)
        response = await self.make_request("GET", "/api/service_b/data/consistency_test")
        assert response.json()["value"] == "updated"
```

#### 3.4 負載和併發測試
```python
# 建立 tests/integration/test_load_and_concurrency.py
class TestLoadAndConcurrency:
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """測試併發請求處理"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # 創建多個併發請求
        concurrent_requests = 50
        
        async def make_test_request(request_id):
            response = await self.make_request(
                "POST", "/api/test_endpoint",
                {"request_id": request_id, "data": "test_data"}
            )
            return response.status_code, response.json()
        
        # 並發執行請求
        tasks = [make_test_request(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 驗證所有請求都成功
        successful_requests = [r for r in results if isinstance(r, tuple) and r[0] == 200]
        assert len(successful_requests) == concurrent_requests
        
        # 驗證沒有數據競爭
        request_ids = [r[1]["processed_request_id"] for r in successful_requests]
        assert len(set(request_ids)) == concurrent_requests  # 所有 ID 都是唯一的
    
    @pytest.mark.asyncio
    async def test_message_queue_under_load(self):
        """測試消息隊列在負載下的表現"""
        from src.common.rabbitmq_client import RabbitMQClient
        
        client = RabbitMQClient()
        await client.connect()
        
        # 發送大量消息
        message_count = 1000
        messages_sent = []
        
        for i in range(message_count):
            message = {"id": i, "data": f"test_message_{i}"}
            await client.publish("load_test_queue", message)
            messages_sent.append(message)
        
        # 接收並驗證消息
        received_messages = []
        
        async def message_handler(message):
            received_messages.append(message)
            if len(received_messages) >= message_count:
                return True  # 停止接收
        
        await client.subscribe("load_test_queue", message_handler)
        
        # 等待所有消息處理完成
        timeout = 30  # 30 秒超時
        start_time = asyncio.get_event_loop().time()
        
        while len(received_messages) < message_count:
            if asyncio.get_event_loop().time() - start_time > timeout:
                break
            await asyncio.sleep(0.1)
        
        assert len(received_messages) == message_count
        
        # 驗證消息順序和內容
        for i, message in enumerate(received_messages):
            assert message["id"] == i
            assert message["data"] == f"test_message_{i}"
```

#### 3.5 實際環境測試腳本
```bash
# 建立 scripts/integration_test.sh
#!/bin/bash

echo "🚀 開始系統整合測試..."

# 步驟 1: 檢查基礎設施
echo "📡 檢查基礎設施服務..."
./scripts/check_infrastructure.sh
if [ $? -ne 0 ]; then
    echo "❌ 基礎設施檢查失敗"
    exit 1
fi

# 步驟 2: 啟動所有服務
echo "🔧 啟動應用服務..."
./scripts/start_all_services.sh
sleep 10  # 等待服務啟動

# 步驟 3: 執行連接測試
echo "🔗 測試服務間連接..."
uv run pytest tests/integration/test_service_connectivity.py -v
if [ $? -ne 0 ]; then
    echo "❌ 服務連接測試失敗"
    ./scripts/stop_all_services.sh
    exit 1
fi

# 步驟 4: 執行端到端工作流測試  
echo "🌊 測試端到端工作流..."
uv run pytest tests/integration/test_end_to_end_workflows.py -v
if [ $? -ne 0 ]; then
    echo "❌ 端到端測試失敗"
    ./scripts/stop_all_services.sh
    exit 1
fi

# 步驟 5: 執行負載測試
echo "⚡ 測試系統負載..."
uv run pytest tests/integration/test_load_and_concurrency.py -v
if [ $? -ne 0 ]; then
    echo "❌ 負載測試失敗" 
    ./scripts/stop_all_services.sh
    exit 1
fi

# 步驟 6: 健康檢查
echo "💚 執行最終健康檢查..."
curl -f http://localhost:8000/health || {
    echo "❌ 健康檢查失敗"
    ./scripts/stop_all_services.sh
    exit 1
}

echo "✅ 所有整合測試通過！"

# 預期：所有測試通過，服務健康運行
# 如果失敗：檢查特定測試的錯誤輸出，修復後重新運行
```

## 最終驗證檢查清單

### 代碼品質檢查
- [ ] 所有單元測試通過：`uv run pytest tests/unit/ -v`
- [ ] 沒有 linting 錯誤：`uv run ruff check src/`
- [ ] 沒有型別錯誤：`uv run mypy src/`
- [ ] 代碼覆蓋率達標：`uv run pytest --cov=src tests/ --cov-report=html`

### 系統整合檢查
- [ ] 基礎設施連接正常：Redis、RabbitMQ、數據庫等服務可訪問
- [ ] 服務間通信測試通過：`uv run pytest tests/integration/test_service_connectivity.py -v`
- [ ] 端到端工作流測試通過：`uv run pytest tests/integration/test_end_to_end_workflows.py -v`
- [ ] 負載和併發測試通過：`uv run pytest tests/integration/test_load_and_concurrency.py -v`
- [ ] 完整整合測試腳本成功：`./scripts/integration_test.sh`

### 功能驗證檢查
- [ ] 手動測試成功：[具體的 curl/命令]
- [ ] 錯誤情況得到優雅處理（網絡中斷、服務不可用等）
- [ ] 數據一致性驗證：跨服務數據同步正常
- [ ] 安全性檢查：認證、授權、數據驗證正常

### 運維就緒檢查  
- [ ] 健康檢查端點響應正常：`curl -f http://localhost:8000/health`
- [ ] 日誌有資訊性但不冗長，包含追蹤 ID
- [ ] 監控指標暴露正常（如使用 Prometheus）
- [ ] 如需要則更新文件和 API 規格
- [ ] 部署腳本和配置文件完整

---

## 要避免的反模式

### 代碼實作反模式
- ❌ 當現有模式有效時不要建立新模式
- ❌ 不要因為"應該可以工作"而跳過驗證
- ❌ 不要忽略失敗的測試 - 修復它們
- ❌ 不要在 async 上下文中使用 sync 函數
- ❌ 不要硬編碼應該是配置的值
- ❌ 不要捕獲所有異常 - 要具體

### 系統整合反模式
- ❌ 不要假設外部服務總是可用 - 實作重試和回退機制
- ❌ 不要忽略服務間的延遲 - 設置適當的超時
- ❌ 不要在測試中使用真實的外部服務 - 使用測試替身
- ❌ 不要忽略數據競爭條件 - 使用適當的同步機制
- ❌ 不要忽略消息順序問題 - 考慮消息的順序性需求
- ❌ 不要忽略事務邊界 - 確保數據一致性
- ❌ 不要在整合測試中測試單元邏輯 - 分層測試責任
