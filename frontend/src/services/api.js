const configuredApiBase = import.meta.env.VITE_API_BASE_URL || '/api';
const API_BASE = configuredApiBase.replace(/\/+$/, '');
const REQUEST_TIMEOUT_MS = {
  default: 30000,
  story: 300000,
  voice: 600000,
  image: 600000,
  audio: 930000,
  video: 1850000,
  movieStatus: 15000,
};

export class ApiError extends Error {
  constructor(message, { status = null, detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function detailText(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    return value
      .map((item) => item?.msg || item?.message || '')
      .filter(Boolean)
      .join('; ');
  }
  return '';
}

async function request(path, options = {}, fallbackMessage = 'The request could not be completed.') {
  const { timeoutMs = REQUEST_TIMEOUT_MS.default, signal, ...fetchOptions } = options;
  const controller = new AbortController();
  let timedOut = false;
  const abortFromCaller = () => controller.abort();
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener('abort', abortFromCaller, { once: true });
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...fetchOptions,
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === 'AbortError' && !timedOut) throw error;
    if (timedOut) {
      throw new ApiError(
        'The request timed out. The provider may still be processing; check its status before retrying.',
        { detail: fallbackMessage },
      );
    }
    throw new ApiError(
      'CineAI could not reach the backend. Check that the API is running and try again.',
      { detail: error.message },
    );
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener('abort', abortFromCaller);
  }

  const contentType = response.headers.get('content-type') || '';
  let payload = null;
  if (contentType.includes('application/json')) {
    try {
      payload = await response.json();
    } catch {
      throw new ApiError('The backend returned an unreadable response.', {
        status: response.status,
      });
    }
  } else {
    payload = await response.text();
  }

  if (!response.ok) {
    const detail = detailText(payload?.detail) || detailText(payload);
    const message = response.status < 500 && detail ? detail : fallbackMessage;
    throw new ApiError(message, { status: response.status, detail: detail || payload });
  }
  return payload;
}

function jsonOptions(body, signal) {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  };
}

export function getErrorDetails(error) {
  if (!import.meta.env.DEV || !error) return null;
  return error.detail ? String(error.detail) : error.message;
}

export function resolveMediaUrl(path) {
  if (!path) return null;
  if (/^https?:\/\//i.test(path)) return path;

  const mediaBase = import.meta.env.VITE_MEDIA_BASE_URL;
  if (mediaBase) return new URL(path, mediaBase).toString();
  if (/^https?:\/\//i.test(API_BASE)) return new URL(path, API_BASE).toString();
  return path.startsWith('/') ? path : `/${path}`;
}

export async function analyzeStory(prompt, genre = 'Cinematic Drama', targetDuration = 1, signal) {
  return request(
    '/story/analyze',
    { ...jsonOptions({
      prompt,
      genre: genre || 'Cinematic Drama',
      target_duration_minutes: targetDuration,
    }, signal), timeoutMs: REQUEST_TIMEOUT_MS.story },
    'Story analysis failed. Review the story text and try again.',
  );
}

export async function getStory(storyId, signal) {
  return request(`/story/${encodeURIComponent(storyId)}`, { signal }, 'The story could not be loaded.');
}

export async function getSampleStories(signal) {
  return request('/story/sample', { signal }, 'Sample stories could not be loaded.');
}

export async function getCharacters(storyId, signal) {
  const query = storyId ? `?story_id=${encodeURIComponent(storyId)}` : '';
  return request(`/characters/${query}`, { signal }, 'Characters could not be loaded.');
}

export async function getScenes(storyId, signal) {
  const query = storyId ? `?story_id=${encodeURIComponent(storyId)}` : '';
  return request(`/scenes/${query}`, { signal }, 'Scenes could not be loaded.');
}

export async function getConfiguredProviders(signal) {
  return request(
    '/providers',
    { signal, timeoutMs: REQUEST_TIMEOUT_MS.default },
    'Provider configuration is unavailable.',
  );
}

export async function generateVoicePreview(characterId, text, signal) {
  return request(
    `/characters/${encodeURIComponent(characterId)}/voice?text=${encodeURIComponent(text)}`,
    { method: 'POST', signal, timeoutMs: REQUEST_TIMEOUT_MS.voice },
    'Voice preview generation failed. Check the configured voice provider.',
  );
}

export async function generateStoryboard(sceneNumber, storyId, promptOverride, signal) {
  return request(
    '/scenes/storyboard',
    { ...jsonOptions({
      scene_number: sceneNumber,
      story_id: storyId,
      prompt_override: promptOverride || null,
    }, signal), timeoutMs: REQUEST_TIMEOUT_MS.image },
    `Storyboard generation failed for scene ${sceneNumber}.`,
  );
}

export async function exportScreenplay(title, storyId, signal) {
  const query = new URLSearchParams({ title, story_id: storyId });
  return request(`/scenes/screenplay/export?${query}`, { signal }, 'Screenplay details could not be loaded.');
}

export async function generateSceneAudio(sceneNumber, options, signal) {
  return request(
    `/scenes/${sceneNumber}/audio`,
    { ...jsonOptions(options, signal), timeoutMs: REQUEST_TIMEOUT_MS.audio },
    `Audio generation failed for scene ${sceneNumber}. Check the configured audio provider.`,
  );
}

export async function generateSceneVideo(sceneNumber, payload, signal) {
  return request(
    `/scenes/${sceneNumber}/video`,
    { ...jsonOptions(payload, signal), timeoutMs: REQUEST_TIMEOUT_MS.video },
    `Video generation failed for scene ${sceneNumber}. Check the configured video provider.`,
  );
}

export async function renderMovie({ storyId, title, fps, resolution }, signal) {
  return request(
    '/movie/render',
    { ...jsonOptions({ story_id: storyId, title, fps, resolution }, signal), timeoutMs: REQUEST_TIMEOUT_MS.default },
    'Movie rendering could not be started. Check the story assets and try again.',
  );
}

export async function checkRenderStatus(movieId, signal) {
  return request(
    `/movie/status/${encodeURIComponent(movieId)}`,
    { signal, timeoutMs: REQUEST_TIMEOUT_MS.movieStatus },
    'Movie status could not be retrieved.',
  );
}
