"""认证模块：API Key 验证 + JWT 签发/验证"""
from datetime import datetime, timedelta, UTC

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import settings

security = HTTPBearer()


def parse_api_keys() -> dict[str, str]:
    """解析配置 api_keys 格式 key1:user1,key2:user2"""
    pairs = [p.strip().split(":") for p in settings.api_keys.split(",") if p.strip()]
    return {k.strip(): v.strip() for k, v in pairs}


# APIKey 专用依赖
async def get_user_by_apikey(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    api_map = parse_api_keys()
    key = credentials.credentials
    user_id = api_map.get(key)
    if not user_id:
        raise HTTPException(status_code=401, detail="无效API Key")
    return user_id


# JWT 工具类
class JWTAuth:
    def __init__(self):
        self.secret = settings.jwt_secret
        self.expire_min = settings.jwt_expire_minutes

    def create_token(self, user_id: str) -> str:
        now = datetime.now(UTC)
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=self.expire_min)
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def verify_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="令牌已过期")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="无效JWT令牌")


# JWT 专用依赖
async def get_user_by_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = credentials.credentials
    jwt_util = JWTAuth()
    payload = jwt_util.verify_token(token)
    return payload["sub"]


# 双模式自适应依赖（自动识别APIKey / JWT）
async def get_current_user_auto(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    cred = credentials.credentials
    api_dict = parse_api_keys()

    # 优先匹配API Key
    if cred in api_dict:
        return api_dict[cred]
    # 不是APIKey则尝试解析JWT
    try:
        jwt_util = JWTAuth()
        payload = jwt_util.verify_token(cred)
        return payload["sub"]
    except HTTPException:
        raise HTTPException(status_code=401, detail="凭证非法：非有效API Key或JWT")
