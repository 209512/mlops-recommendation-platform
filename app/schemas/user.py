from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """사용자 기본 스키마"""

    email: EmailStr
    username: str
    nickname: str | None = None
    is_active: bool = True


class UserCreate(UserBase):
    """사용자 생성 스키마"""

    password: str


class UserUpdate(BaseModel):
    """사용자 업데이트 스키마"""

    nickname: str | None = None
    is_active: bool | None = None


class User(UserBase):
    """사용자 응답 스키마"""

    id: int
    last_login: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
