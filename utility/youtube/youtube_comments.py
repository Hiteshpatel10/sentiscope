import pandas as pd
import os


def extract_video_id(youtube_url):
    video_id = youtube_url.split("v=")[-1].split("&")[0]
    return video_id

def get_comments(service, video_ID, uuid):
    comments = []
    response = service.commentThreads().list(part="snippet", videoId=video_ID, textFormat="plainText").execute()
    page = 0
    while len(comments) <= 1000:
        page += 1
        index = 0
        for item in response['items']:
            index += 1
            comment = item["snippet"]["topLevelComment"]
            text = comment["snippet"]["textDisplay"]
            date = comment["snippet"]["publishedAt"]
            author = comment["snippet"]["authorDisplayName"]
            like_count = comment["snippet"]["likeCount"]
            reply_count = item["snippet"]["totalReplyCount"]
            comments.append((text, date, author, like_count, reply_count))
        if 'nextPageToken' in response:
            response = service.commentThreads().list(part="snippet", videoId=video_ID, textFormat="plainText", pageToken=response['nextPageToken']).execute()
        else:
            break
        
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
 
    filename = f"{uuid}.csv"
    output_df = pd.DataFrame.from_dict(comments)
    output_df.to_csv(f'./data/{filename}', index=False, header=['comment', 'date', 'author', 'like_count', 'reply_count'])
