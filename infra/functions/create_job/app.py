"""POST /jobs {lat, lon, profile, content_type} -> {job_id, upload_url} (presigned PUT, 5 min)."""

import json
import os
import uuid
from datetime import datetime, timezone

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "ap-south-1")
TABLE = os.environ["TABLE_NAME"]
BUCKET = os.environ["BUCKET_NAME"]
PROFILES = {"school", "worker", "general"}
EXTENSIONS = {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov"}
URL_EXPIRY_S = 300

s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)


def _resp(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def _validate(body: dict) -> tuple[float, float, str, str]:
    try:
        lat, lon = float(body["lat"]), float(body["lon"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("lat and lon must be numbers")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat must be in [-90, 90] and lon in [-180, 180]")
    profile, content_type = body.get("profile"), body.get("content_type")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    if content_type not in EXTENSIONS:
        raise ValueError(f"content_type must be one of {sorted(EXTENSIONS)}")
    return lat, lon, profile, content_type


def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        lat, lon, profile, content_type = _validate(body)
    except (json.JSONDecodeError, ValueError) as e:
        return _resp(400, {"error": str(e)})

    job_id = uuid.uuid4().hex
    key = f"uploads/{job_id}.{EXTENSIONS[content_type]}"
    # Numbers as strings: boto3's DynamoDB serializer rejects floats.
    table.put_item(Item={"job_id": job_id, "status": "pending", "lat": str(lat), "lon": str(lon), "profile": profile,
                         "created_at": datetime.now(timezone.utc).isoformat(), "s3_key": key})
    url = s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type},
                                    ExpiresIn=URL_EXPIRY_S)
    return _resp(201, {"job_id": job_id, "upload_url": url})
