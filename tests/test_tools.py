from gaia_agent.tools import web_search


def test_web_search_returns_results_text():
    result = web_search("Python programming language")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "python" in result.lower()


from gaia_agent.tools import fetch_page


def test_fetch_page_extracts_text():
    result = fetch_page("https://example.com")
    assert isinstance(result, str)
    assert "Example Domain" in result


from gaia_agent.tools import download_gaia_file


def test_download_gaia_file_handles_missing_file_gracefully():
    result = download_gaia_file("nonexistent-task-id-12345")
    assert isinstance(result, str)
    assert result.startswith("ERROR")


from gaia_agent.tools import read_file


def test_read_file_plain_text(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("hello world")
    result = read_file(str(p))
    assert result == "hello world"


def test_read_file_missing_returns_error():
    result = read_file("/nonexistent/path/file.txt")
    assert result.startswith("ERROR")


from gaia_agent.tools import python_exec


def test_python_exec_prints_output():
    result = python_exec("print(2 + 2)")
    assert "4" in result


def test_python_exec_captures_errors():
    result = python_exec("raise ValueError('boom')")
    assert "boom" in result


def test_python_exec_times_out_long_running_code():
    result = python_exec("import time; time.sleep(60)")
    assert "TIMEOUT" in result


from gaia_agent.tools import transcribe_audio


def test_transcribe_audio_returns_text_for_real_speech():
    result = transcribe_audio("/usr/share/sounds/speech-dispatcher/test.wav")
    assert isinstance(result, str)
    assert not result.startswith("ERROR")
    assert len(result) > 0


def test_transcribe_audio_missing_file_returns_error():
    result = transcribe_audio("/nonexistent/path/audio.mp3")
    assert result.startswith("ERROR")


from gaia_agent.tools import transcribe_youtube_video, analyze_youtube_frames


def test_transcribe_youtube_video_returns_dialogue_text():
    # Real network test against a short, stable public video -- matches the
    # pattern of other tests in this file that hit real services rather
    # than mocking, since the whole point is verifying the real pipeline
    # (yt-dlp download + ffmpeg audio extraction + Whisper) works together.
    result = transcribe_youtube_video("https://www.youtube.com/watch?v=1htKBjuUWec")
    assert isinstance(result, str)
    assert not result.startswith("ERROR")
    assert "extremely" in result.lower()


def test_transcribe_youtube_video_invalid_url_returns_error():
    result = transcribe_youtube_video("https://www.youtube.com/watch?v=nonexistent_xyz123")
    assert result.startswith("ERROR")


def test_analyze_youtube_frames_invalid_url_returns_error():
    result = analyze_youtube_frames("https://www.youtube.com/watch?v=nonexistent_xyz123")
    assert result.startswith("ERROR")
