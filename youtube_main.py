from utility.youtube.youtube_comments import get_comments, extract_video_id
from utility.youtube.youtube_senti import senti 
from googleapiclient.discovery import build

def youtubeMain(videoId, uuid):
    api_service_name = "youtube"
    api_version = "v3"
    api_key = "AIzaSyAOvxr2XvdsjlqFnc-E0EgKbzlEfXb88_4" 
    yt_service = build(api_service_name, api_version, developerKey=api_key)

    get_comments(yt_service, videoId, uuid)
    senti(uuid)

if __name__ == "__main__":
    youtubeMain()

