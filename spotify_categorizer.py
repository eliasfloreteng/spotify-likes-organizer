import os
import time
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import openai
from tqdm import tqdm
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for required environment variables
missing_vars = []
if not SPOTIFY_CLIENT_ID:
    missing_vars.append("SPOTIFY_CLIENT_ID")
if not SPOTIFY_CLIENT_SECRET:
    missing_vars.append("SPOTIFY_CLIENT_SECRET")
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")

if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please create a .env file with these variables")
    exit(1)

# Spotify API scopes needed - reduced scope as we're only reading
SCOPE = "user-library-read"

# LLM batch size and parameters
SONGS_PER_BATCH = 20  # Number of songs to categorize in one LLM call
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Output file paths
LIKED_SONGS_FILE = "spotify_liked_songs.json"
SONG_CATEGORIES_FILE = "song_categories.json"
CATEGORIZATION_SUMMARY_FILE = "categorization_summary.json"

# OpenAI API configuration
openai.api_key = OPENAI_API_KEY
LLM_MODEL = os.getenv(
    "OPENAI_MODEL", "gpt-3.5-turbo"
)  # Allow customizing the model via env var


def setup_spotify_client():
    """Set up and return an authenticated Spotify client."""
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope=SCOPE,
        )
    )
    return sp


def get_all_liked_songs(sp):
    """Fetch all liked songs from Spotify in batches."""
    logger.info("Fetching liked songs from Spotify...")

    # Check if we already have the songs cached
    if os.path.exists(LIKED_SONGS_FILE):
        logger.info(f"Loading liked songs from cache: {LIKED_SONGS_FILE}")
        with open(LIKED_SONGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    offset = 0
    limit = 50  # Spotify API limit for this endpoint
    all_songs = []

    while True:
        results = sp.current_user_saved_tracks(limit=limit, offset=offset)
        if not results["items"]:
            break

        for item in results["items"]:
            track = item["track"]

            # Get all artists, not just the first one
            artists = ", ".join([artist["name"] for artist in track["artists"]])

            # Get additional track features if available
            try:
                song_info = {
                    "id": track["id"],
                    "name": track["name"],
                    "artist": artists,
                    "album": track["album"]["name"],
                    "uri": track["uri"],
                    "popularity": track["popularity"],
                    "added_at": item["added_at"],
                    "release_date": track["album"].get("release_date", "Unknown"),
                }
                all_songs.append(song_info)
            except Exception as e:
                logger.warning(
                    f"Error processing track {track.get('name', 'Unknown')}: {e}"
                )

        offset += limit
        logger.info(f"Fetched {len(all_songs)} songs so far...")

        # Respect Spotify API rate limits
        time.sleep(0.1)

    logger.info(f"Total liked songs: {len(all_songs)}")

    # Save the songs to a cache file
    with open(LIKED_SONGS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_songs, f, indent=2)

    return all_songs


def load_existing_categories():
    """Load existing song categorizations if available."""
    if os.path.exists(SONG_CATEGORIES_FILE):
        logger.info(f"Loading existing categorizations from: {SONG_CATEGORIES_FILE}")
        with open(SONG_CATEGORIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_categories(categories):
    """Save song categorizations to file."""
    # Create a timestamped backup of the existing file if it exists
    if os.path.exists(SONG_CATEGORIES_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{SONG_CATEGORIES_FILE}.{timestamp}.bak"
        try:
            with open(SONG_CATEGORIES_FILE, "r", encoding="utf-8") as src:
                with open(backup_file, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
            logger.info(f"Created backup of categories file: {backup_file}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    # Save the updated categories
    with open(SONG_CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)
    logger.info(f"Saved categories to: {SONG_CATEGORIES_FILE}")


def categorize_songs_with_llm(songs_batch, existing_categories=None):
    """
    Use LLM to categorize a batch of songs into multiple categories.
    Returns a list of lists, where each inner list contains categories for a song.
    """
    # Format the songs for the prompt
    songs_text = "\n".join(
        [
            f"{i + 1}. '{song['name']}' by {song['artist']} (Album: {song['album']})"
            for i, song in enumerate(songs_batch)
        ]
    )

    # Add information about existing categories to encourage reuse
    existing_categories_text = ""
    if existing_categories and len(existing_categories) > 0:
        # Extract unique categories from existing categorizations
        all_cats = []
        for song_cats in existing_categories.values():
            all_cats.extend(song_cats)

        unique_cats = sorted(list(set(all_cats)))

        if unique_cats:
            existing_categories_text = (
                "Existing playlist categories (reuse these when appropriate):\n"
            )
            existing_categories_text += ", ".join(
                unique_cats[:50]
            )  # Limit to prevent token overflow
            existing_categories_text += "\n\n"

    prompt = f"""{existing_categories_text}Categorize these songs into music genres, mood-based playlists, or other relevant groupings.
For each song, assign 2-4 categories that best describe it. Use common genre names, moods, eras, or themes.
Return the results in this format:
1. Category A | Category B | Category C
2. Category D | Category E | Category F
And so on. Just the category names in order, separated by pipes (|), nothing else.

Songs to categorize:
{songs_text}"""

    # Retry mechanism for API calls
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a music categorization expert who organizes songs into playlist categories.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent categorization
            )

            # Parse the categories from the response
            categories_text = response.choices[0].message.content.strip().split("\n")
            all_categories = []

            for line in categories_text:
                if ". " in line:
                    line = line.split(". ", 1)[1]  # Remove the number prefix
                categories = [cat.strip() for cat in line.split("|")]
                all_categories.append(categories)

            # Make sure we have categories for each song
            if len(all_categories) >= len(songs_batch):
                return all_categories[: len(songs_batch)]
            else:
                logger.warning(
                    f"LLM returned {len(all_categories)} category sets for {len(songs_batch)} songs. Retrying..."
                )
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Error in LLM call: {e}. Attempt {attempt + 1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)

    # If all retries fail, return a default category
    return [["Uncategorized"]] * len(songs_batch)


def generate_summary(song_categories, all_songs):
    """Generate a summary of categorizations."""
    logger.info("Generating categorization summary...")

    # Create a lookup dict for songs by ID
    songs_by_id = {song["id"]: song for song in all_songs}

    # Count songs per category
    categories = {}
    for song_id, song_cats in song_categories.items():
        for category in song_cats:
            if category not in categories:
                categories[category] = {"count": 0, "songs": []}

            if song_id in songs_by_id:
                song_info = songs_by_id[song_id]
                categories[category]["count"] += 1
                categories[category]["songs"].append(
                    {
                        "id": song_id,
                        "name": song_info["name"],
                        "artist": song_info["artist"],
                        "uri": song_info["uri"],
                    }
                )

    # Sort categories by song count
    sorted_categories = {
        k: v
        for k, v in sorted(
            categories.items(), key=lambda item: item[1]["count"], reverse=True
        )
    }

    # Create the summary
    summary = {
        "total_songs_categorized": len(song_categories),
        "total_categories": len(sorted_categories),
        "categories": sorted_categories,
    }

    # Save the summary
    with open(CATEGORIZATION_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {CATEGORIZATION_SUMMARY_FILE}")
    logger.info(
        f"Found {len(sorted_categories)} unique categories across {len(song_categories)} songs"
    )

    # Print top categories
    top_cats = list(sorted_categories.items())[:15]
    logger.info("Top 15 categories:")
    for cat, data in top_cats:
        logger.info(f"  {cat}: {data['count']} songs")

    return summary


def main():
    # Setup
    logger.info("Starting Spotify music categorizer")
    sp = setup_spotify_client()

    # Get all liked songs
    all_songs = get_all_liked_songs(sp)

    # Load existing categorizations if available
    song_categories = load_existing_categories()

    # Filter songs that haven't been categorized yet
    uncategorized_songs = [
        song for song in all_songs if song["id"] not in song_categories
    ]
    logger.info(
        f"Found {len(uncategorized_songs)} uncategorized songs out of {len(all_songs)} total"
    )

    # Process songs in batches
    if uncategorized_songs:
        for i in tqdm(
            range(0, len(uncategorized_songs), SONGS_PER_BATCH),
            desc="Categorizing songs",
        ):
            batch = uncategorized_songs[i : i + SONGS_PER_BATCH]
            categories = categorize_songs_with_llm(batch, song_categories)

            # Save categories for each song
            for j, song in enumerate(batch):
                if j < len(categories):
                    song_categories[song["id"]] = categories[j]

            # Save progress after each batch
            save_categories(song_categories)

            # Sleep to respect API rate limits
            time.sleep(1)
    else:
        logger.info("No new songs to categorize")

    # Generate and save categorization summary
    generate_summary(song_categories, all_songs)

    logger.info("Music categorization complete! Results saved to disk.")


if __name__ == "__main__":
    main()
