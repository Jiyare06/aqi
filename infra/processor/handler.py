"""S3 ObjectCreated on uploads/{job_id}.<ext> -> frames, causes, air, advice, audio -> result_json in DynamoDB.

Never writes to uploads/ (that would re-trigger this function). Any error -> status=failed + error.
causes.py: ONNX smoke/fire + CAM overlay -> overlays/; advice.py: Bedrock via llm.py, template fallback;
audio.py: Polly HI + EN -> audio/.
"""

import json
import os
import traceback
import urllib.parse
from pathlib import Path

import boto3
from botocore.config import Config

import advice
import audio
import causes
import frames
from air.combine import air_for

REGION = os.environ.get("AWS_REGION", "ap-south-1")
TABLE = os.environ["TABLE_NAME"]
BUCKET = os.environ.get("BUCKET_NAME", "")

s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)
# The ONNX session is loaded lazily by causes.detect() on the first invocation, outside the 10 s init phase.


def _set(job_id: str, **fields) -> None:
    names = {f"#{k}": k for k in fields}  # "status" is a DynamoDB reserved word
    values = {f":{k}": v for k, v in fields.items()}
    table.update_item(Key={"job_id": job_id}, UpdateExpression="SET " + ", ".join(f"#{k} = :{k}" for k in fields),
                      ExpressionAttributeNames=names, ExpressionAttributeValues=values)


def job_id_from_key(key: str) -> str:
    if not key.startswith("uploads/"):
        raise ValueError(f"unexpected key {key!r}")
    return Path(key).stem


def process(bucket: str, key: str, job_id: str) -> dict:
    job = table.get_item(Key={"job_id": job_id}).get("Item")
    if job is None:
        raise ValueError(f"no job row for {job_id}")
    _set(job_id, status="processing")
    lat, lon, profile = float(job["lat"]), float(job["lon"]), job["profile"]

    clip = Path("/tmp") / Path(key).name
    try:
        s3.download_file(bucket, key, str(clip))
        sampled = frames.sample(clip)
    finally:
        clip.unlink(missing_ok=True)

    air = air_for(lat, lon, os.environ.get("WAQI_TOKEN") or None, profile=profile)
    found, cause_caveats, model_ran = causes.analyse(sampled, job_id, s3, BUCKET)
    text, sections = advice.build(air, profile, found, model_ran)
    spoken, audio_caveats = audio.synthesize(sections, job_id)
    return {
        "job_id": job_id,
        "status": "done",
        "created_at": job["created_at"],
        "profile": profile,
        "location": {"lat": lat, "lon": lon, "place": None},  # no reverse geocoding yet
        "air": air["air"],
        "forecast_today": air["forecast_today"],
        "forecast_tomorrow": air["forecast_tomorrow"],
        "decision": air["decision"],
        "seasonal_line": None,  # seasonal lookup not wired yet
        "causes": found,
        "advice": text,
        "audio": spoken,
        "caveats": air["caveats"] + cause_caveats + audio_caveats,
    }


def handler(event, context):
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        job_id = None
        try:
            job_id = job_id_from_key(key)
            result = process(bucket, key, job_id)
            _set(job_id, status="done", result_json=json.dumps(result, ensure_ascii=False))
            print(json.dumps({"job_id": job_id, "status": "done", "frames_used": result["causes"]["frames_used"]}))
        except Exception as e:
            traceback.print_exc()
            if job_id:
                try:
                    _set(job_id, status="failed", error=f"{type(e).__name__}: {e}")
                except Exception:
                    traceback.print_exc()
