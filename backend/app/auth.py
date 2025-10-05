import os
from fastapi import HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt

SECRET = os.getenv('JWT_SECRET', 'dev-secret')


class LoginPayload(BaseModel):
    username: str
    password: str


def create_token(username: str):
    exp = datetime.utcnow() + timedelta(hours=8)
    payload = {"sub": username, "exp": exp.isoformat()}
    token = jwt.encode(payload, SECRET, algorithm='HS256')
    return token


async def login(payload: LoginPayload = Depends()):
    # This is a placeholder: normally validate against DB
    if payload.username != 'admin' or payload.password != 'password':
        raise HTTPException(status_code=401, detail='Invalid credentials')
    return {"access_token": create_token(payload.username), "token_type": "bearer"}
