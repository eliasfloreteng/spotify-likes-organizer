import os
import time
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import openai
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SPOTIFY_CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
SPOTIFY_CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Spotify API scopes needed
SCOPE = "user-library-read playlist-modify-private playlist-read-private"

# LLM batch size and parameters
SONGS_PER_BATCH = 20  # Number of songs to categorize in one LLM call
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
PLAYLIST_CACHE_FILE = "playlist_cache.json"
SONG_CATEGORIES_FILE = "song_categories.json"

# OpenAI API configuration
openai.api_key = OPENAI_API_KEY
LLM_MODEL = "gpt-3.5-turbo"  # More economical than GPT-4

def setup_spotify_client():
    """Set up and return an authenticated Spotify client."""
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    ))
    return sp

def get_all_liked_songs(sp):
    """Fetch all liked songs from Spotify in batches."""
    logger.info("Fetching liked songs from Spotify...")
    
    offset = 0
    limit = 50  # Spotify API limit for this endpoint
    all_songs = []
    
    while True:
        results = sp.current_user_saved_tracks(limit=limit, offset=offset)
        if not results['items']:
            break
            
        for item in results['items']:
            track = item['track']
            song_info = {
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'uri': track['uri']
            }
            all_songs.append(song_info)
            
        offset += limit
        logger.info(f"Fetched {len(all_songs)} songs so far...")
        
        # Respect Spotify API rate limits
        time.sleep(0.1)
    
    logger.info(f"Total liked songs: {len(all_songs)}")
    return all_songs

def load_existing_categories():
    """Load existing song categorizations if available."""
    if os.path.exists(SONG_CATEGORIES_FILE):
        with open(SONG_CATEGORIES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_categories(categories):
    """Save song categorizations to file."""
    with open(SONG_CATEGORIES_FILE, 'w') as f:
        json.dump(categories, f)

def categorize_songs_with_llm(songs_batch):
    """Use LLM to categorize a batch of songs."""
    # Format the songs for the prompt
    songs_text = "\n".join([f"{i+1}. '{song['name']}' by {song['artist']} (Album: {song['album']})" 
                          for i, song in enumerate(songs_batch)])
    
    prompt = f"""Categorize these songs into music genres or mood-based playlists. 
For each song, assign ONE category only. Be specific but use common genre names or moods.
Return the results in this format:
1. Category Name
2. Category Name
And so on. Just the category names in order, nothing else.

Songs to categorize:
{songs_text}"""

    # Retry mechanism for API calls
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a music categorization expert who organizes songs into playlist categories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent categorization
            )
            
            # Parse the categories from the response
            categories_text = response.choices[0].message.content.strip().split('\n')
            categories = [line.split('. ')[1] if '. ' in line else line for line in categories_text]
            
            # Make sure we have a category for each song
            if len(categories) >= len(songs_batch):
                return categories[:len(songs_batch)]
            else:
                logger.warning(f"LLM returned {len(categories)} categories for {len(songs_batch)} songs. Retrying...")
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Error in LLM call: {e}. Attempt {attempt+1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)
    
    # If all retries fail, return a default category
    return ["Uncategorized"] * len(songs_batch)

def create_or_get_playlist(sp, name, playlist_cache):
    """Create a new playlist or get existing one with the given name."""
    if name in playlist_cache:
        return playlist_cache[name]
    
    # Check if playlist already exists
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        if playlist['name'] == name:
            playlist_cache[name] = playlist['id']
            return playlist['id']
    
    # Create new playlist
    user_id = sp.me()['id']
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=False,
        description=f"Auto-generated playlist: {name}"
    )
    playlist_cache[name] = playlist['id']
    return playlist['id']

def save_playlist_cache(playlist_cache):
    """Save playlist cache to file."""
    with open(PLAYLIST_CACHE_FILE, 'w') as f:
        json.dump(playlist_cache, f)

def load_playlist_cache():
    """Load playlist cache from file."""
    if os.path.exists(PLAYLIST_CACHE_FILE):
        with open(PLAYLIST_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def add_songs_to_playlists(sp, song_categories, playlist_cache):
    """Add songs to their respective playlists."""
    # Group songs by category
    categories = {}
    for song_id, category in song_categories.items():
        if category not in categories:
            categories[category] = []
        categories[category].append(song_id)
    
    # Add songs to each playlist
    for category, song_ids in tqdm(categories.items(), desc="Creating playlists"):
        playlist_id = create_or_get_playlist(sp, category, playlist_cache)
        
        # Add songs in batches (Spotify API limit)
        batch_size = 100
        for i in range(0, len(song_ids), batch_size):
            batch = song_ids[i:i+batch_size]
            try:
                sp.playlist_add_items(playlist_id, batch)
                time.sleep(0.5)  # Respect API rate limits
            except Exception as e:
                logger.error(f"Error adding songs to {category} playlist: {e}")
    
    save_playlist_cache(playlist_cache)

def main():
    # Setup
    logger.info("Starting Spotify playlist organizer")
    sp = setup_spotify_client()
    
    # Get all liked songs
    all_songs = get_all_liked_songs(sp)
    
    # Load existing categorizations if available
    song_categories = load_existing_categories()
    
    # Filter songs that haven't been categorized yet
    uncategorized_songs = [song for song in all_songs if song['id'] not in song_categories]
    logger.info(f"Found {len(uncategorized_songs)} uncategorized songs")
    
    # Process songs in batches
    for i in tqdm(range(0, len(uncategorized_songs), SONGS_PER_BATCH), desc="Categorizing songs"):
        batch = uncategorized_songs[i:i+SONGS_PER_BATCH]
        categories = categorize_songs_with_llm(batch)
        
        # Save categories for each song
        for j, song in enumerate(batch):
            if j < len(categories):
                song_categories[song['id']] = categories[j]
        
        # Save progress after each batch
        save_categories(song_categories)
        
        # Sleep to respect API rate limits
        time.sleep(1)
    
    # Create playlists and add songs
    logger.info("Creating playlists and adding songs")
    playlist_cache = load_playlist_cache()
    add_songs_to_playlists(sp, song_categories, playlist_cache)
    
    logger.info("Playlist organization complete!")

if __name__ == "__main__":
    main()