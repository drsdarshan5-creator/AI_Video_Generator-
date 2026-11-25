import { useEffect, useRef, useState } from 'react';
import {
  analyzeStory,
  checkRenderStatus,
  exportScreenplay,
  generateSceneAudio,
  generateSceneVideo,
  generateStoryboard,
  generateVoicePreview,
  getConfiguredProviders,
  getErrorDetails,
  renderMovie,
  resolveMediaUrl,
} from './services/api';

const STEPS = [
  { id: 1, title: 'Story', subtitle: 'Premise & analysis' },
  { id: 2, title: 'Characters', subtitle: 'Cast & voice profiles' },
  { id: 3, title: 'Scenes', subtitle: 'Storyboard & timeline' },
  { id: 4, title: 'Generate', subtitle: 'Assets & final movie' },
];

const PROVIDER_LABELS = {
  image: {
    placeholder: 'Placeholder images',
    onnx_directml: 'AI Image Generation',
  },
  voice: {
    placeholder: 'Placeholder voice',
    piper_onnx: 'Piper ONNX',
  },
  video: {
    placeholder: 'Placeholder video',
    cogvideox: 'CogVideoX',
  },
  audio: {
    placeholder: 'Placeholder audio',
    local_audio: 'Procedural Audio',
    ai_audio: 'AudioLDM 2',
  },
};

const POLL_INTERVAL_MS = Math.max(
  1000,
  Number(import.meta.env.VITE_MOVIE_POLL_INTERVAL_MS) || 2500,
);
const POLL_TIMEOUT_MS = Math.max(
  POLL_INTERVAL_MS,
  Number(import.meta.env.VITE_MOVIE_POLL_TIMEOUT_MS) || 30 * 60 * 1000,
);

function providerLabel(kind, provider) {
  if (!provider) return 'Checking configuration';
  return PROVIDER_LABELS[kind]?.[provider] || 'Custom provider';
}

function analyzerLabel(provider) {
  if (provider === 'rule_based') return 'Rule-based analysis';
  if (provider === 'local_llm') return 'Local story model';
  return 'Configured analyzer';
}

function audioTypeLabel(audioType) {
  return {
    ambience: 'Ambience',
    sound_effect: 'Sound effect',
    music: 'Music',
  }[audioType] || 'Ambience';
}

function sceneAudioSuggestion(scene) {
  const action = (scene.action_description || '').toLowerCase();
  const environment = [
    scene.heading,
    scene.location,
    scene.weather,
    scene.action_description,
  ].filter(Boolean).join(' ').toLowerCase();
  if (/\b(footsteps?|steps?|running|run|door|slam|explosion|blast|vehicle|car|truck|engine|impact|crash|gunfire|knock|splash)\b/.test(action)) {
    return 'sound_effect';
  }
  if (/\b(rain|rainfall|wind|storm|city|urban|street|forest|woods|crowd|market|ocean|sea|waves|river|birds|traffic|room tone)\b/.test(environment)) {
    return 'ambience';
  }
  if (/\b(dramatic|tense|suspense|romantic|emotional|hopeful|melancholy|eerie|mysterious|uplifting|joyful)\b/.test(scene.mood || '')) {
    return 'music';
  }
  return 'ambience';
}

function durationLabel(seconds) {
  const safeSeconds = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function movieStage(job) {
  if (job?.status === 'completed') return 'Completed';
  if (job?.status === 'failed') return 'Failed';
  const message = (job?.message || '').toLowerCase();
  if (message.includes('audio assets')) return 'Generating audio';
  if (message.includes('synchroniz')) return 'Synchronizing audio';
  if (message.includes('validat')) return 'Validating output';
  if (message.includes('initializ') || message.includes('prepar')) return 'Preparing';
  if (message.includes('encoding') || message.includes('assembling')) return 'Assembling movie';
  return 'Processing';
}

function responseAssetState(response, kind) {
  const status = response?.status || (response?.[`${kind}_url`] ? 'generated' : 'failed');
  const rawMessage = response?.error || response?.message || null;
  const failureMessages = {
    image: 'Storyboard generation failed. Check the configured image provider.',
    video: 'Scene video generation failed. Check the configured video provider.',
    audio: 'Audio generation failed. Check the configured audio provider.',
  };
  const message = status === 'failed'
    ? failureMessages[kind]
    : status === 'placeholder'
      ? 'The configured provider is a placeholder; no media file was created.'
      : rawMessage;
  return {
    status,
    url: response?.[`${kind}_url`] || null,
    prompt: response?.visual_prompt || response?.metadata?.prompt || null,
    provider: response?.provider_name || response?.provider || null,
    message,
    error: rawMessage ? { message: rawMessage, detail: rawMessage } : null,
  };
}

function ErrorDetails({ error }) {
  const details = getErrorDetails(error);
  if (!details) return null;
  return (
    <details className="technical-details">
      <summary>Technical details</summary>
      <pre>{details}</pre>
    </details>
  );
}

export default function App() {
  const [currentStep, setCurrentStep] = useState(1);
  const [titleInput, setTitleInput] = useState('');
  const [storyText, setStoryText] = useState('');
  const [genre, setGenre] = useState('');
  const [tone, setTone] = useState('');
  const [language, setLanguage] = useState('');
  const [targetDuration, setTargetDuration] = useState(3);
  const [storyData, setStoryData] = useState(null);
  const [screenplay, setScreenplay] = useState(null);
  const [screenplayError, setScreenplayError] = useState('');
  const [storyError, setStoryError] = useState(null);
  const [storyLoading, setStoryLoading] = useState(false);
  const [providers, setProviders] = useState(null);
  const [providerError, setProviderError] = useState(null);
  const [sceneAssets, setSceneAssets] = useState({});
  const [voiceAssets, setVoiceAssets] = useState({});
  const [audioTypes, setAudioTypes] = useState({});
  const [movieTitle, setMovieTitle] = useState('');
  const [movieResolution, setMovieResolution] = useState('1080p');
  const [movieFps, setMovieFps] = useState(24);
  const [movieJob, setMovieJob] = useState(null);
  const [movieError, setMovieError] = useState(null);
  const [moviePollError, setMoviePollError] = useState('');
  const [moviePollTimedOut, setMoviePollTimedOut] = useState(false);
  const [movieStarting, setMovieStarting] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    getConfiguredProviders(controller.signal)
      .then(setProviders)
      .catch((error) => {
        if (error.name !== 'AbortError') setProviderError(error);
      });
    return () => controller.abort();
  }, []);

  const movieId = movieJob?.movie_id;
  const movieStatus = movieJob?.status;

  useEffect(() => {
    if (!movieId || movieStatus !== 'processing') return undefined;

    let active = true;
    let inFlight = false;
    let timerId;
    let controller;
    const startedAt = Date.now();

    const poll = async () => {
      if (!active || inFlight) return;
      if (Date.now() - startedAt >= POLL_TIMEOUT_MS) {
        setMoviePollTimedOut(true);
        return;
      }

      inFlight = true;
      let terminalStatus = false;
      controller = new AbortController();
      try {
        const latest = await checkRenderStatus(movieId, controller.signal);
        if (active) {
          setMovieJob(latest);
          setMoviePollError('');
        }
        if (latest.status === 'completed' || latest.status === 'failed') {
          terminalStatus = true;
          return;
        }
      } catch (error) {
        if (active && error.name !== 'AbortError') {
          setMoviePollError(error.message || 'Movie status could not be retrieved.');
        }
      } finally {
        inFlight = false;
        if (active && !terminalStatus) {
          if (Date.now() - startedAt >= POLL_TIMEOUT_MS) {
            setMoviePollTimedOut(true);
          } else {
            timerId = window.setTimeout(poll, POLL_INTERVAL_MS);
          }
        }
      }
    };

    timerId = window.setTimeout(poll, POLL_INTERVAL_MS);
    return () => {
      active = false;
      window.clearTimeout(timerId);
      controller?.abort();
    };
  }, [movieId, movieStatus]);

  const updateSceneAsset = (sceneNumber, kind, asset) => {
    setSceneAssets((current) => ({
      ...current,
      [sceneNumber]: {
        ...current[sceneNumber],
        [kind]: { ...current[sceneNumber]?.[kind], ...asset },
      },
    }));
  };

  const updateSceneRecord = (sceneNumber, updates) => {
    setStoryData((current) => {
      if (!current) return current;
      return {
        ...current,
        scenes: current.scenes.map((scene) =>
          scene.scene_number === sceneNumber ? { ...scene, ...updates } : scene,
        ),
      };
    });
  };

  const handleAnalyze = async (event) => {
    event.preventDefault();
    if (storyText.trim().length < 10) {
      setStoryError({ message: 'Enter at least 10 characters of story text to analyze.' });
      return;
    }

    setStoryLoading(true);
    setStoryError(null);
    setScreenplayError('');
    setScreenplay(null);
    const guidance = [
      storyText.trim(),
      tone.trim() ? `Preferred narrative tone: ${tone.trim()}.` : '',
      language.trim() ? `Preferred story language: ${language.trim()}.` : '',
    ].filter(Boolean).join('\n\n');

    try {
      const result = await analyzeStory(
        guidance,
        genre.trim() || 'Cinematic Drama',
        Number(targetDuration) || 1,
      );
      setStoryData(result);
      setMovieJob(null);
      setMovieError(null);
      setMoviePollError('');
      setMoviePollTimedOut(false);
      setMovieTitle(titleInput.trim() || result.title);
      setSceneAssets(Object.fromEntries((result.scenes || []).map((scene) => [
        scene.scene_number,
        {
          storyboard: scene.image_url
          ? { status: 'existing', url: scene.image_url }
            : { status: 'pending', url: null },
          video: scene.video_url
          ? { status: 'existing', url: scene.video_url }
            : { status: 'pending', url: null },
          audio: { status: 'pending', url: null },
        },
      ])));
      setVoiceAssets({});
      setAudioTypes(Object.fromEntries((result.scenes || []).map((scene) => [
        scene.scene_number,
        sceneAudioSuggestion(scene),
      ])));
      setCurrentStep(2);
      try {
        const screenplayResult = await exportScreenplay(
          titleInput.trim() || result.title,
          result.story_id,
        );
        setScreenplay(screenplayResult);
      } catch (error) {
        setScreenplayError(error.message || 'Screenplay details could not be loaded.');
      }
    } catch (error) {
      setStoryError({
        message: error.message || 'Story analysis failed. Review the story text and try again.',
        error,
      });
    } finally {
      setStoryLoading(false);
    }
  };

  const handleGenerateStoryboard = async (scene) => {
    updateSceneAsset(scene.scene_number, 'storyboard', { status: 'generating', message: '' });
    try {
      const response = await generateStoryboard(
        scene.scene_number,
        storyData.story_id,
        scene.visual_prompt,
      );
      const asset = responseAssetState(response, 'image');
      if (asset.status === 'failed' || !asset.url) {
        updateSceneAsset(scene.scene_number, 'storyboard', {
          ...asset,
          status: 'failed',
          url: null,
          message: asset.message || 'The storyboard provider returned no image.',
        });
        return;
      }
      updateSceneAsset(scene.scene_number, 'storyboard', asset);
      updateSceneRecord(scene.scene_number, { image_url: asset.url });
    } catch (error) {
      updateSceneAsset(scene.scene_number, 'storyboard', {
        status: 'failed',
        url: null,
        message: error.message || 'Storyboard generation failed.',
        error,
      });
    }
  };

  const handleGenerateAllStoryboards = async () => {
    if (!storyData) return;
    for (const scene of storyData.scenes) {
      await handleGenerateStoryboard(scene);
    }
  };

  const handleVoicePreview = async (character) => {
    const dialogue = (storyData?.scenes || [])
      .flatMap((scene) => scene.dialogues || [])
      .find((line) => line.character_name?.toLowerCase() === character.name.toLowerCase());
    const text = dialogue?.text || `This is a voice preview for ${character.name}.`;
    setVoiceAssets((current) => ({
      ...current,
      [character.id]: { status: 'generating', url: null },
    }));
    try {
      const response = await generateVoicePreview(character.id, text);
      const status = response.status || (response.audio_url ? 'generated' : 'placeholder');
      setVoiceAssets((current) => ({
        ...current,
        [character.id]: {
          status,
          url: response.audio_url || null,
          message: status === 'failed'
            ? response.message || 'Voice preview generation failed.'
            : status === 'placeholder'
              ? response.message || 'The configured voice provider did not return an audio file.'
              : '',
        },
      }));
    } catch (error) {
      setVoiceAssets((current) => ({
        ...current,
        [character.id]: {
          status: 'failed',
          url: null,
          message: error.message || 'Voice preview generation failed.',
          error,
        },
      }));
    }
  };

  const handleGenerateVideo = async (scene) => {
    updateSceneAsset(scene.scene_number, 'video', { status: 'generating', message: '' });
    try {
      const payload = {
        scene_id: scene.id ? String(scene.id) : `${storyData.story_id}:${scene.scene_number}`,
        story_id: storyData.story_id,
        prompt: scene.visual_prompt || scene.action_description,
        seed: scene.scene_number,
        metadata: { heading: scene.heading },
      };
      if (providers?.video && providers.video !== 'cogvideox' && scene.duration_seconds) {
        payload.duration_seconds = scene.duration_seconds;
      }
      const response = await generateSceneVideo(scene.scene_number, payload);
      const asset = responseAssetState(response, 'video');
      if (asset.status === 'generated' && asset.url) {
        updateSceneRecord(scene.scene_number, { video_url: asset.url });
      }
      updateSceneAsset(scene.scene_number, 'video', asset);
    } catch (error) {
      updateSceneAsset(scene.scene_number, 'video', {
        status: 'failed',
        url: null,
        message: error.message || 'Video generation failed.',
        error,
      });
    }
  };

  const handleGenerateAllVideos = async () => {
    if (!storyData) return;
    for (const scene of storyData.scenes) {
      await handleGenerateVideo(scene);
    }
  };

  const handleGenerateAudio = async (scene) => {
    const audioType = audioTypes[scene.scene_number] || sceneAudioSuggestion(scene);
    updateSceneAsset(scene.scene_number, 'audio', { status: 'generating', message: '' });
    try {
      const response = await generateSceneAudio(scene.scene_number, {
        story_id: storyData.story_id,
        audio_type: audioType,
        duration_seconds: scene.duration_seconds,
        seed: scene.scene_number,
      });
      const asset = responseAssetState(response, 'audio');
      updateSceneAsset(scene.scene_number, 'audio', asset);
    } catch (error) {
      updateSceneAsset(scene.scene_number, 'audio', {
        status: 'failed',
        url: null,
        message: error.message || 'Audio generation failed.',
        error,
      });
    }
  };

  const handleStartMovie = async () => {
    if (!storyData?.story_id || !storyData.scenes?.length) {
      setMovieError({
        message: 'Analyze a story with at least one scene before assembling a movie.',
      });
      return;
    }
    const missingVisuals = storyData.scenes
      .filter((scene) => !scene.video_url && !scene.image_url)
      .map((scene) => scene.scene_number);
    if (missingVisuals.length) {
      setMovieError({
        message: `Generate a storyboard image or scene video for each scene before assembling the movie. Missing scenes: ${missingVisuals.join(', ')}.`,
      });
      return;
    }

    setMovieStarting(true);
    setMovieError(null);
    setMoviePollError('');
    setMoviePollTimedOut(false);
    try {
      const job = await renderMovie({
        storyId: storyData.story_id,
        title: movieTitle.trim() || storyData.title,
        fps: Number(movieFps),
        resolution: movieResolution,
      });
      setMovieJob(job);
      setCurrentStep(4);
    } catch (error) {
      setMovieError({
        message: error.message || 'Movie rendering could not be started.',
        error,
      });
    } finally {
      setMovieStarting(false);
    }
  };

  const storyScenes = storyData?.scenes || [];
  const storyCharacters = storyData?.characters || [];
  const movieDuration = storyScenes.reduce(
    (total, scene) => total + (Number(scene.duration_seconds) || 0),
    0,
  );
  const isMovieActive = movieJob?.status === 'processing';

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="CineAI Studio home">
          <span className="brand-mark" aria-hidden="true">C</span>
          <span>
            <strong>CineAI</strong>
            <small>Story-to-screen studio</small>
          </span>
        </a>
        <div className="topbar-meta">
          <span className={`connection-indicator ${providers ? 'is-connected' : ''}`} />
          <span>{providers ? 'Backend connected' : 'Backend status unknown'}</span>
          <span className="environment-badge">Local workspace</span>
        </div>
      </header>

      <main className="workspace">
        <section className="workspace-heading">
          <div>
            <p className="eyebrow">CINEMATIC PRODUCTION WORKSPACE</p>
            <h1>From story idea to final cut.</h1>
            <p className="workspace-intro">
              Analyze a premise, review the scenes, generate assets only when you choose,
              then assemble a movie from the available media.
            </p>
          </div>
          {storyData && (
            <div className="project-chip">
              <span className="project-chip-label">CURRENT STORY</span>
              <strong>{movieTitle || storyData.title}</strong>
              <span>{storyScenes.length} scenes · {storyCharacters.length} characters</span>
            </div>
          )}
        </section>

        {providerError && (
          <div className="notice notice-warning" role="status">
            Provider configuration is unavailable. You can still use the workflow; provider
            names will appear once the backend responds.
            <ErrorDetails error={providerError} />
          </div>
        )}

        <nav className="workflow-nav" aria-label="Production workflow">
          {STEPS.map((step) => (
            <button
              key={step.id}
              type="button"
              className={`workflow-step ${currentStep === step.id ? 'is-active' : ''} ${storyData && step.id < currentStep ? 'is-complete' : ''}`}
              onClick={() => setCurrentStep(step.id)}
              disabled={step.id > 1 && !storyData}
              aria-current={currentStep === step.id ? 'step' : undefined}
            >
              <span className="step-number">{storyData && step.id < currentStep ? '✓' : `0${step.id}`}</span>
              <span className="step-copy"><strong>{step.title}</strong><small>{step.subtitle}</small></span>
            </button>
          ))}
        </nav>

        {currentStep === 1 && (
          <section className="panel story-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">01 / STORY DEVELOPMENT</p>
                <h2>Start with a story</h2>
                <p>Give CineAI a premise to analyze. No media generation starts until you request it.</p>
              </div>
              <span className="step-pill">Analysis only</span>
            </div>
            <form onSubmit={handleAnalyze} className="story-form">
              <div className="form-grid">
                <label className="field">
                  <span>Working title <small>Optional</small></span>
                  <input
                    value={titleInput}
                    onChange={(event) => setTitleInput(event.target.value)}
                    placeholder="The name of your film"
                    maxLength={120}
                  />
                </label>
                <label className="field">
                  <span>Genre <small>Optional</small></span>
                  <input
                    value={genre}
                    onChange={(event) => setGenre(event.target.value)}
                    placeholder="e.g. Science fiction"
                    maxLength={120}
                  />
                </label>
                <label className="field">
                  <span>Tone <small>Optional guidance</small></span>
                  <select value={tone} onChange={(event) => setTone(event.target.value)}>
                    <option value="">Let the story guide the tone</option>
                    <option value="dramatic">Dramatic</option>
                    <option value="suspenseful">Suspenseful</option>
                    <option value="hopeful">Hopeful</option>
                    <option value="romantic">Romantic</option>
                    <option value="comedic">Comedic</option>
                  </select>
                </label>
                <label className="field">
                  <span>Language <small>Optional guidance</small></span>
                  <input
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                    placeholder="e.g. English"
                    maxLength={60}
                  />
                </label>
                <label className="field field-duration">
                  <span>Target duration</span>
                  <select value={targetDuration} onChange={(event) => setTargetDuration(event.target.value)}>
                    <option value={1}>1 minute</option>
                    <option value={3}>3 minutes</option>
                    <option value={5}>5 minutes</option>
                    <option value={10}>10 minutes</option>
                  </select>
                </label>
              </div>
              <label className="field story-field">
                <span>Story premise <b aria-label="required">*</b></span>
                <textarea
                  value={storyText}
                  onChange={(event) => setStoryText(event.target.value)}
                  rows={8}
                  minLength={10}
                  required
                  placeholder="Describe the central character, their goal, and the conflict that changes everything..."
                />
                <small className="field-hint">At least 10 characters. Tone and language guidance are included with the premise.</small>
              </label>
              {storyError && (
                <div className="notice notice-error" role="alert">
                  <strong>{storyError.message}</strong>
                  {storyError.error && <ErrorDetails error={storyError.error} />}
                </div>
              )}
              <div className="form-footer">
                <p>Story analysis uses the configured backend analyzer. Asset generation remains a separate, explicit step.</p>
                <button className="button button-primary" type="submit" disabled={storyLoading || storyText.trim().length < 10}>
                  {storyLoading ? <><span className="button-spinner" /> Analyzing story…</> : 'Analyze story'}
                  {!storyLoading && <span aria-hidden="true">→</span>}
                </button>
              </div>
            </form>
          </section>
        )}

        {currentStep === 2 && storyData && (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">02 / CAST & STORY</p>
                <h2>Story overview</h2>
                <p>Review the analyzed story and character profiles before generating assets.</p>
              </div>
              <button type="button" className="button button-quiet" onClick={() => setCurrentStep(1)}>Edit story</button>
            </div>
            <div className="story-summary">
              <div className="summary-title">
                <span className="summary-kicker">{storyData.genre || 'Cinematic story'} · {storyData.tone || 'Tone not specified'}</span>
                <h3>{storyData.title}</h3>
                <p>{storyData.logline}</p>
              </div>
              <div className="summary-columns">
                {storyData.synopsis && <div><h4>Synopsis</h4><p>{storyData.synopsis}</p></div>}
                {storyData.theme && <div><h4>Theme</h4><p>{storyData.theme}</p></div>}
              </div>
              <div className="summary-meta">
                <span>Story ID <code>{storyData.story_id}</code></span>
                <span>Analyzer <b>{analyzerLabel(storyData.provider)}</b></span>
                <span>{storyScenes.length} scenes</span>
              </div>
            </div>

            <div className="section-heading">
              <div><p className="eyebrow">CHARACTER INTELLIGENCE</p><h3>Cast profiles</h3></div>
              <span className="count-badge">{storyCharacters.length} characters</span>
            </div>
            {storyCharacters.length ? (
              <div className="character-grid">
                {storyCharacters.map((character) => {
                  const voice = voiceAssets[character.id];
                  return (
                    <article className="character-card" key={character.id}>
                      <div className="character-card-top">
                        <div className="avatar-mark">{character.name?.slice(0, 1)?.toUpperCase() || 'C'}</div>
                        <div><span className="character-role">{character.role || 'Character'}</span><h4>{character.name}</h4></div>
                      </div>
                      <dl className="detail-list">
                        <div><dt>Age range</dt><dd>{character.age_range || 'Not specified'}</dd></div>
                        <div><dt>Personality</dt><dd>{character.personality || 'Not specified'}</dd></div>
                        <div><dt>Appearance</dt><dd>{character.appearance_description || character.description || 'Not specified'}</dd></div>
                        <div><dt>Clothing</dt><dd>{character.clothing_style || 'Not specified'}</dd></div>
                        <div><dt>Voice profile</dt><dd>{character.voice_name || 'Default voice'}</dd></div>
                      </dl>
                      {character.visual_prompt && <details className="prompt-details"><summary>Visual prompt</summary><p>{character.visual_prompt}</p></details>}
                      <div className="asset-action-row">
                        <button
                          className="button button-secondary button-small"
                          type="button"
                          onClick={() => handleVoicePreview(character)}
                          disabled={voice?.status === 'generating'}
                        >
                          {voice?.status === 'generating' ? 'Generating preview…' : 'Generate voice preview'}
                        </button>
                        <span className={`status-tag status-${voice?.status || 'idle'}`}>
                          {voice?.status || 'Not generated'}
                        </span>
                      </div>
                      {voice?.url && <audio className="audio-player" controls src={resolveMediaUrl(voice.url)}>Audio preview unavailable.</audio>}
                      {voice?.message && <p className="asset-message">{voice.message}</p>}
                      {voice?.error && <ErrorDetails error={voice.error} />}
                    </article>
                  );
                })}
              </div>
            ) : <p className="empty-state">No character profiles were returned for this story.</p>}

            <div className="panel-footer">
              <div className="screenplay-info">
                <strong>Screenplay</strong>
                {screenplay
                  ? <span>{screenplay.scenes?.length || storyScenes.length} formatted scenes · {screenplay.formatted_text ? 'formatted screenplay available' : 'scene outline available'}</span>
                  : <span>{screenplayError || 'Screenplay details are not available.'}</span>}
              </div>
              <button type="button" className="button button-primary" onClick={() => setCurrentStep(3)}>
                Review scenes <span aria-hidden="true">→</span>
              </button>
            </div>
          </section>
        )}

        {currentStep === 3 && storyData && (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">03 / SCENE DESIGN</p>
                <h2>Scenes & storyboard</h2>
                <p>Inspect the timeline and generate storyboard frames explicitly, scene by scene.</p>
              </div>
              <div className="panel-heading-actions">
                <span className="count-badge">{storyScenes.length} scenes · {durationLabel(movieDuration)} estimated</span>
              </div>
            </div>
            {storyScenes.length ? (
              <div className="scene-list">
                {storyScenes.map((scene) => {
                  const storyboardAsset = sceneAssets[scene.scene_number]?.storyboard || { status: 'pending' };
                  const hasFrame = storyboardAsset.url && storyboardAsset.status !== 'failed';
                  const dialogue = scene.dialogues || [];
                  return (
                    <article className="scene-card" key={scene.scene_number}>
                      <div className="scene-frame">
                        {hasFrame ? (
                          <img src={resolveMediaUrl(storyboardAsset.url)} alt={`Storyboard for scene ${scene.scene_number}`} />
                        ) : (
                          <div className="frame-empty">
                            <span className="frame-icon">▧</span>
                            <strong>{storyboardAsset.status === 'failed' ? 'Frame generation failed' : 'No storyboard frame'}</strong>
                            <span>Generate a frame when you are ready</span>
                          </div>
                        )}
                        <span className={`status-tag frame-status status-${storyboardAsset.status || 'pending'}`}>
                          {storyboardAsset.status || 'pending'}
                        </span>
                      </div>
                      <div className="scene-content">
                        <div className="scene-title-row">
                          <div><span className="scene-number">SCENE {String(scene.scene_number).padStart(2, '0')}</span><h3>{scene.heading}</h3></div>
                          <span className="duration-pill">{scene.duration_seconds || 5}s</span>
                        </div>
                        <div className="scene-facts">
                          <span><b>Location</b>{scene.location || 'Not specified'}</span>
                          <span><b>Time</b>{scene.time_of_day || 'Not specified'}</span>
                          <span><b>Weather</b>{scene.weather || 'Not specified'}</span>
                          <span><b>Mood</b>{scene.mood || 'Not specified'}</span>
                          <span><b>Lighting</b>{scene.lighting || 'Not specified'}</span>
                          <span><b>Audio suggestion</b>{audioTypeLabel(audioTypes[scene.scene_number] || sceneAudioSuggestion(scene))}</span>
                        </div>
                        <p className="scene-action">{scene.action_description}</p>
                        {dialogue.length > 0 && (
                          <div className="dialogue-list">
                            <strong>Dialogue</strong>
                            {dialogue.map((line, index) => (
                              <p key={`${scene.scene_number}-${index}`}><b>{line.character_name}</b><span>“{line.text}”</span></p>
                            ))}
                          </div>
                        )}
                        {scene.visual_prompt && <details className="prompt-details"><summary>Visual prompt</summary><p>{scene.visual_prompt}</p></details>}
                        {storyboardAsset.prompt && <p className="generated-prompt">Frame prompt: {storyboardAsset.prompt}</p>}
                        {storyboardAsset.message && <p className="asset-message">{storyboardAsset.message}</p>}
                        {storyboardAsset.error && <ErrorDetails error={storyboardAsset.error} />}
                        <div className="asset-action-row scene-actions">
                          <button
                            type="button"
                            className="button button-secondary button-small"
                            disabled={storyboardAsset.status === 'generating'}
                            onClick={() => handleGenerateStoryboard(scene)}
                          >
                            {storyboardAsset.status === 'generating' ? 'Generating frame…' : hasFrame ? 'Regenerate frame' : 'Generate storyboard'}
                          </button>
                          <span className={`provider-caption ${storyboardAsset.provider ? '' : 'muted'}`}>
                            {storyboardAsset.provider ? providerLabel('image', storyboardAsset.provider) : providerLabel('image', providers?.image)}
                          </span>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : <p className="empty-state">The analyzer did not return any scenes.</p>}
            <div className="panel-footer">
              <p className="footnote">Movie assembly requires each scene to have a storyboard image or a scene video. Missing assets are checked before the render starts.</p>
              <button type="button" className="button button-primary" onClick={() => setCurrentStep(4)}>
                Generation controls <span aria-hidden="true">→</span>
              </button>
            </div>
          </section>
        )}

        {currentStep === 4 && storyData && (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">04 / PRODUCTION & DELIVERY</p>
                <h2>Generation controls</h2>
                <p>Choose individual assets to generate. GPU-backed image, video, and audio providers run only after an explicit request.</p>
              </div>
              <span className="step-pill">On demand</span>
            </div>

            <div className="provider-strip" aria-label="Configured providers">
              {['image', 'voice', 'video', 'audio'].map((kind) => (
                <div className="provider-item" key={kind}>
                  <span>{kind}</span>
                  <strong>{providerLabel(kind, providers?.[kind])}</strong>
                  <small>{providers ? 'Configured on backend' : 'Configuration unavailable'}</small>
                </div>
              ))}
            </div>

            <div className="generation-grid">
              <article className="generation-card generation-card-wide">
                <div className="generation-card-heading">
                  <div><span className="generation-icon image-icon">01</span><div><h3>Storyboard images</h3><p>Generate one frame per scene using the configured image provider.</p></div></div>
                  <div className="generation-heading-actions">
                    <span className="provider-caption">{providerLabel('image', providers?.image)}</span>
                    <button
                      type="button"
                      className="text-button"
                      disabled={storyScenes.some((scene) => sceneAssets[scene.scene_number]?.storyboard?.status === 'generating')}
                      onClick={handleGenerateAllStoryboards}
                    >
                      Generate all
                    </button>
                  </div>
                </div>
                <div className="generation-list">
                  {storyScenes.map((scene) => {
                    const asset = sceneAssets[scene.scene_number]?.storyboard || { status: 'pending' };
                    return (
                      <div className="generation-row" key={`image-${scene.scene_number}`}>
                        <span>Scene {scene.scene_number}</span>
                        <span className={`status-tag status-${asset.status}`}>{asset.status}</span>
                        <button type="button" className="text-button" disabled={asset.status === 'generating'} onClick={() => handleGenerateStoryboard(scene)}>
                          {asset.url ? 'Regenerate' : 'Generate'}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </article>

              <article className="generation-card">
                <div className="generation-card-heading">
                  <div><span className="generation-icon voice-icon">02</span><div><h3>Dialogue voice</h3><p>Preview a representative line for each character.</p></div></div>
                  <span className="provider-caption">{providerLabel('voice', providers?.voice)}</span>
                </div>
                <div className="generation-list">
                  {storyCharacters.map((character) => {
                    const asset = voiceAssets[character.id] || { status: 'pending' };
                    return (
                      <div className="generation-row" key={`voice-${character.id}`}>
                        <span>{character.name}</span>
                        <span className={`status-tag status-${asset.status}`}>{asset.status}</span>
                        <button type="button" className="text-button" disabled={asset.status === 'generating'} onClick={() => handleVoicePreview(character)}>Preview</button>
                        {asset.url && <audio className="audio-player row-audio" controls src={resolveMediaUrl(asset.url)}>Audio preview unavailable.</audio>}
                      </div>
                    );
                  })}
                </div>
              </article>

              <article className="generation-card">
                <div className="generation-card-heading">
                  <div><span className="generation-icon video-icon">03</span><div><h3>Scene video</h3><p>Generate clips explicitly. Provider limits and availability apply.</p></div></div>
                  <div className="generation-heading-actions">
                    <span className="provider-caption">{providerLabel('video', providers?.video)}</span>
                    <button
                      type="button"
                      className="text-button"
                      disabled={storyScenes.some((scene) => sceneAssets[scene.scene_number]?.video?.status === 'generating')}
                      onClick={handleGenerateAllVideos}
                    >
                      Generate all
                    </button>
                  </div>
                </div>
                <div className="generation-list">
                  {storyScenes.map((scene) => {
                    const asset = sceneAssets[scene.scene_number]?.video || { status: 'pending' };
                    return (
                      <div className="generation-row" key={`video-${scene.scene_number}`}>
                        <span>Scene {scene.scene_number}</span>
                        <span className={`status-tag status-${asset.status}`}>{asset.status}</span>
                        <button type="button" className="text-button" disabled={asset.status === 'generating'} onClick={() => handleGenerateVideo(scene)}>
                          {asset.url ? 'Regenerate' : 'Generate'}
                        </button>
                        {asset.url && <video className="row-video" controls playsInline preload="metadata" src={resolveMediaUrl(asset.url)} />}
                        {asset.message && <p className="asset-message row-message">{asset.message}</p>}
                        {asset.error && <ErrorDetails error={asset.error} />}
                      </div>
                    );
                  })}
                </div>
              </article>

              <article className="generation-card">
                <div className="generation-card-heading">
                  <div><span className="generation-icon audio-icon">04</span><div><h3>Scene audio</h3><p>Generate ambience, effects, or music for a selected scene.</p></div></div>
                  <span className="provider-caption">{providerLabel('audio', providers?.audio)}</span>
                </div>
                <div className="generation-list">
                  {storyScenes.map((scene) => {
                    const asset = sceneAssets[scene.scene_number]?.audio || { status: 'pending' };
                    const selectedType = audioTypes[scene.scene_number] || sceneAudioSuggestion(scene);
                    return (
                      <div className="generation-row audio-generation-row" key={`audio-${scene.scene_number}`}>
                        <span>Scene {scene.scene_number}</span>
                        <select
                          aria-label={`Scene ${scene.scene_number} audio type`}
                          value={selectedType}
                          onChange={(event) => setAudioTypes((current) => ({ ...current, [scene.scene_number]: event.target.value }))}
                        >
                          <option value="ambience">Ambience</option>
                          <option value="sound_effect">Sound effect</option>
                          <option value="music">Music</option>
                        </select>
                        <span className={`status-tag status-${asset.status}`}>{asset.status}</span>
                        <button type="button" className="text-button" disabled={asset.status === 'generating'} onClick={() => handleGenerateAudio(scene)}>
                          {asset.url ? 'Regenerate' : 'Generate'}
                        </button>
                        {asset.url && <audio className="audio-player row-audio" controls src={resolveMediaUrl(asset.url)}>Audio playback unavailable.</audio>}
                        {asset.message && <p className="asset-message row-message">{asset.message}</p>}
                        {asset.error && <ErrorDetails error={asset.error} />}
                      </div>
                    );
                  })}
                </div>
              </article>
            </div>

            <section className="movie-section" aria-labelledby="movie-heading">
              <div className="section-heading movie-heading">
                <div><p className="eyebrow">FINAL ASSEMBLY</p><h3 id="movie-heading">Generate movie</h3></div>
                <span className="provider-caption">Uses existing scene assets</span>
              </div>
              <div className="movie-options">
                <label className="field">
                  <span>Movie title</span>
                  <input value={movieTitle} onChange={(event) => setMovieTitle(event.target.value)} maxLength={120} />
                </label>
                <label className="field">
                  <span>Output resolution</span>
                  <select value={movieResolution} onChange={(event) => setMovieResolution(event.target.value)}>
                    <option value="480p">480p</option><option value="720p">720p</option><option value="1080p">1080p</option>
                  </select>
                </label>
                <label className="field">
                  <span>Frame rate</span>
                  <select value={movieFps} onChange={(event) => setMovieFps(event.target.value)}>
                    {[24, 25, 30].map((fps) => <option value={fps} key={fps}>{fps} fps</option>)}
                  </select>
                </label>
                <div className="movie-estimate"><span>Estimated timeline</span><strong>{durationLabel(movieDuration)}</strong><small>{storyScenes.length} scenes · duration follows scene settings</small></div>
              </div>
              {movieError && (
                <div className="notice notice-error" role="alert">
                  <strong>{movieError.message}</strong>
                  {movieError.error && <ErrorDetails error={movieError.error} />}
                </div>
              )}
              <div className="movie-action">
                <p>Movie assembly will not start until you click Generate movie. Missing scene visuals are checked before submission.</p>
                <button className="button button-primary" type="button" onClick={handleStartMovie} disabled={movieStarting || isMovieActive}>
                  {movieStarting ? <><span className="button-spinner" /> Starting render…</> : isMovieActive ? 'Render in progress' : 'Generate movie'}
                  {!movieStarting && !isMovieActive && <span aria-hidden="true">→</span>}
                </button>
              </div>
            </section>

            {movieJob && (
              <section className="render-panel" aria-live="polite">
                <div className="render-header">
                  <div>
                    <p className="eyebrow">MOVIE JOB · {movieJob.movie_id}</p>
                    <h3>{movieJob.title || movieTitle || 'Movie render'}</h3>
                  </div>
                  <span className={`status-tag status-${movieJob.status}`}>{movieJob.status}</span>
                </div>
                <div className="render-stage">
                  <div><span>Current stage</span><strong>{movieStage(movieJob)}</strong></div>
                  <span className="render-progress-value">{Number.isFinite(movieJob.progress) ? `${movieJob.progress}%` : 'Progress unavailable'}</span>
                </div>
                {Number.isFinite(movieJob.progress) && (
                  <div
                    className="progress-track"
                    role="progressbar"
                    aria-label="Movie render progress"
                    aria-valuemin="0"
                    aria-valuemax="100"
                    aria-valuenow={movieJob.progress}
                  >
                    <span style={{ width: `${Math.min(100, Math.max(0, movieJob.progress))}%` }} />
                  </div>
                )}
                <p className="render-message">{movieJob.status === 'failed'
                  ? 'Movie assembly failed. Confirm each scene has a generated image or video, then try again.'
                  : movieJob.message || 'Waiting for the backend render status.'}</p>
                {moviePollError && isMovieActive && <p className="notice notice-warning" role="status">Status check failed; polling will retry. {moviePollError}</p>}
                {moviePollTimedOut && isMovieActive && (
                  <p className="notice notice-warning" role="alert">
                    Status polling stopped after the configured wait limit. The backend job may still be running; you can check its status by starting a new session or contacting the operator.
                  </p>
                )}
                {movieJob.status === 'failed' && movieJob.message && import.meta.env.DEV && (
                  <details className="technical-details"><summary>Backend failure details</summary><pre>{movieJob.message}</pre></details>
                )}
                {movieJob.status === 'completed' && movieJob.video_url && (
                  <div className="movie-result">
                    <div className="movie-result-meta">
                      <div><span>Runtime estimate</span><strong>{durationLabel(movieDuration)}</strong></div>
                      <div><span>Requested output</span><strong>{movieResolution} · {movieFps} fps · MP4</strong></div>
                    </div>
                    <video
                      ref={videoRef}
                      className="movie-player"
                      controls
                      playsInline
                      preload="metadata"
                      src={resolveMediaUrl(movieJob.video_url)}
                    >
                      Your browser does not support HTML video.
                    </video>
                    <div className="movie-result-actions">
                      <button type="button" className="button button-primary" onClick={() => videoRef.current?.play()}>
                        Play movie <span aria-hidden="true">▶</span>
                      </button>
                      <a className="button button-secondary" href={resolveMediaUrl(movieJob.video_url)} download>
                        Download MP4 <span aria-hidden="true">↓</span>
                      </a>
                    </div>
                    <p className="media-reference">Output reference <code>{movieJob.video_url}</code></p>
                  </div>
                )}
                {movieJob.status === 'completed' && !movieJob.video_url && (
                  <p className="notice notice-warning">The backend marked the render complete but did not return a playable movie URL.</p>
                )}
              </section>
            )}

            <div className="panel-footer">
              <button type="button" className="button button-quiet" onClick={() => setCurrentStep(3)}>← Back to scenes</button>
              {storyData && <span className="footnote">Assets not generated remain unchanged. Provider calls are never started on page load.</span>}
            </div>
          </section>
        )}
      </main>
      <footer className="app-footer">
        <span>CineAI Studio</span><span>Generation is explicit · Progress is reported by the backend</span>
      </footer>
    </div>
  );
}
