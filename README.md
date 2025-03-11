# Spotify Likes Organizer

A Python tool that fetches your Spotify liked songs and organizes them into categories using AI for easier playlist creation.

## Features

- Fetches all your liked songs from Spotify
- Uses OpenAI's GPT models to categorize songs by genre, mood, era, etc.
- Saves categorization data for future reference
- Generates a summary of categories and song counts
- Caches song data to minimize API calls
- Creates automatic backups of categorization data

## Prerequisites

- Python 3.6+
- Spotify Developer account
- OpenAI API key

## Installation

1. Clone this repository:

```
git clone https://github.com/eliasfloreteng/spotify-likes-organizer.git
cd spotify-likes-organizer
```

2. Install the required dependencies:

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install --upgrade pip
pip install -r requirements.txt
```

3. Create a `.env` file by copying the example:

```
cp .env.example .env
```

## Configuration

### Spotify API Setup

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Log in with your Spotify account
3. Create a new application
4. Set a Redirect URI (e.g., `http://localhost:8888/callback`)
5. Copy your Client ID and Client Secret

### Configure Environment Variables

Edit the `.env` file and add your credentials:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

## Usage

Run the script to fetch and categorize your liked songs:

```
python spotify_categorizer.py
```

The first time you run the script, it will:

1. Open a browser window asking you to authorize the application with your Spotify account
2. Fetch all your liked songs from Spotify
3. Save them to `spotify_liked_songs.json`
4. Categorize the songs using the specified OpenAI model
5. Save the categorizations to `song_categories.json`
6. Generate a summary in `categorization_summary.json`

Subsequent runs will only categorize newly liked songs that weren't previously processed.

## Output Files

- **spotify_liked_songs.json**: Cache of your liked songs from Spotify
- **song_categories.json**: Contains the AI-generated categories for each song
- **categorization_summary.json**: Summary of all categories with song counts

## Limitations

- The OpenAI API has rate limits and usage costs
- The script processes songs in batches to optimize API usage
- Spotify API limits the number of tracks that can be fetched in one request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
