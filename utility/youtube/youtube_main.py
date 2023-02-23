from youtube_comments import get_comments, extract_video_id
from youtube_senti import senti
from googleapiclient.discovery import build
import csv

def main():
    api_service_name = "youtube"
    api_version = "v3"
    api_key = "AIzaSyAOvxr2XvdsjlqFnc-E0EgKbzlEfXb88_4" 
    yt_service = build(api_service_name, api_version, developerKey=api_key)

    videoUrl = "https://www.youtube.com/watch?v=DWtwZJfFBXg&ab_channel=bekifaayati"
    videoId = extract_video_id(videoUrl)
    get_comments(yt_service, videoId)
    senti()

if __name__ == "__main__":
    main()

