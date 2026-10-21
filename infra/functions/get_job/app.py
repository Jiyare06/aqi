"""GET /jobs/{id} -> {job_id, status, result (parsed result_json or null), error, audio_urls, gradcam_url}.

`result` is stored exactly in the contract shape (audio and causes.gradcam_key hold S3 keys). `audio_urls` and
`gradcam_url` = fresh presigned GET URLs (1 h) for those keys, made on every read so nothing stale is stored.
"""

import json
import os

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "ap-south-1")
BUCKET = os.environ["BUCKET_NAME"]
URL_EXPIRY_S = 3600

s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
table = boto3.resource("dynamodb", region_name=REGION).Table(os.environ["TABLE_NAME"])


def _resp(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, ensure_ascii=False)}


def presign(key: str | None) -> str | None:
    if not key:
        return None
    return s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=URL_EXPIRY_S)


def audio_urls(result: dict | None) -> dict:
    audio = (result or {}).get("audio") or {}
    return {lang: presign(audio.get(f"{lang}_key")) for lang in ("hi", "en")}


def gradcam_url(result: dict | None) -> str | None:
    return presign(((result or {}).get("causes") or {}).get("gradcam_key"))


def handler(event, context):
    job_id = (event.get("pathParameters") or {}).get("id", "")
    if not job_id:
        return _resp(400, {"error": "missing job id"})
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        return _resp(404, {"error": "job not found"})
    result = json.loads(item["result_json"]) if item.get("result_json") else None
    return _resp(200, {"job_id": job_id, "status": item["status"], "result": result, "error": item.get("error"),
                       "audio_urls": audio_urls(result), "gradcam_url": gradcam_url(result)})
