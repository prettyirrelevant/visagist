import { pipeline, ImageClassificationPipeline } from '@huggingface/transformers';
import { createConsola } from 'consola/browser';

const log = createConsola({ level: 4 }).withTag('Classify');

export interface ClassificationResult {
  isMeme: boolean;
  confidence: number;
  inferenceTimeMs: number;
}

const HUGGINGFACE_MODEL = 'prettyirrelevant/meme-detector-onnx';

let classifier: ImageClassificationPipeline | null = null;
let modelStatus: 'loading' | 'downloading' | 'ready' | 'error' = 'loading';


/**
 * loads the ml model for classification from hugging face.
 * tries webgpu first for performance, falls back to wasm if unavailable.
 */
export async function loadModel(): Promise<void> {
  log.info('loading model from hugging face');

  if (classifier) {
    log.debug('model already loaded');
    modelStatus = 'ready';
    return;
  }

  try {
    modelStatus = 'downloading';
    log.info('downloading model from hugging face');

    // try webgpu first, fall back to wasm
    try {
      log.debug('attempting webgpu device');
      classifier = await pipeline('image-classification', HUGGINGFACE_MODEL, {
        device: 'webgpu',
        dtype: 'q8',
      });
      modelStatus = 'ready';
      log.info('model loaded successfully with webgpu');
    } catch (webgpuError) {
      log.warn('webgpu failed, falling back to wasm', {
        error: webgpuError instanceof Error ? webgpuError.message : String(webgpuError)
      });

      classifier = await pipeline('image-classification', HUGGINGFACE_MODEL, {
        device: 'wasm',
        dtype: 'q8',
      });
      modelStatus = 'ready';
      log.info('model loaded successfully with wasm');
    }
  } catch (error) {
    log.error('failed to load model', {
      error: error instanceof Error ? error.message : String(error)
    });
    modelStatus = 'error';
    throw new Error(`unable to load model: ${HUGGINGFACE_MODEL}`);
  }
}


/**
 * classifies multiple images in parallel.
 */
export async function classifyImageBatch(imageUrls: string[]): Promise<ClassificationResult[]> {
  if (!classifier) {
    log.error('model not loaded for batch classification');
    throw new Error('model not loaded. call loadModel() first.');
  }

  if (imageUrls.length === 0) {
    log.debug('empty batch provided');
    return [];
  }

  log.info('starting batch classification', { batchSize: imageUrls.length });

  try {
    const classificationPromises = imageUrls.map(url => classifyImage(url));
    const results = await Promise.all(classificationPromises);

    const memeCount = results.filter(r => r.isMeme).length;
    const avgInferenceTimeMs = results.reduce((sum, r) => sum + r.inferenceTimeMs, 0) / results.length;

    log.info('batch classification completed', {
      batchSize: imageUrls.length,
      memeCount,
      avgInferenceTimeMs
    });

    return results;
  } catch (error) {
    log.error('batch classification failed', {
      error: error instanceof Error ? error.message : String(error),
      batchSize: imageUrls.length
    });

    // return default results for all urls
    return imageUrls.map(() => ({
      isMeme: false,
      confidence: 0,
      inferenceTimeMs: 0
    }));
  }
}


/**
 * classifies a single image url as meme or not meme.
 * returns safe defaults on error to maintain system stability.
 */
export async function classifyImage(imageUrl: string): Promise<ClassificationResult> {
  if (!classifier) {
    log.error('model not loaded');
    throw new Error('model not loaded. call loadModel() first.');
  }

  const startTimeMs = performance.now();
  log.debug('starting classification', { imageUrl });

  try {
    const results = await classifier(imageUrl);
    const inferenceTimeMs = performance.now() - startTimeMs;

    // validate results
    if (!Array.isArray(results) || results.length === 0) {
      log.warn('invalid classification result', { results });
      return {
        isMeme: false,
        confidence: 0,
        inferenceTimeMs
      };
    }

    const { isMeme, confidence } = determineMemeClassification(results);

    log.debug('classification completed', {
      imageUrl,
      isMeme,
      confidence,
      inferenceTimeMs,
      results
    });

    return {
      isMeme,
      confidence,
      inferenceTimeMs
    };
  } catch (error) {
    const inferenceTimeMs = performance.now() - startTimeMs;
    log.error('classification failed', {
      error: error instanceof Error ? error.message : String(error),
      imageUrl,
      inferenceTimeMs
    });

    // return safe defaults on error
    return {
      isMeme: false,
      confidence: 0,
      inferenceTimeMs
    };
  }
}


/**
 * gets the current model loading status.
 */
export function getModelStatus(): 'loading' | 'downloading' | 'ready' | 'error' {
  log.debug('model status check', { status: modelStatus });
  return modelStatus;
}

/**
 * checks if the model is loaded and ready for use.
 */
export function isModelReady(): boolean {
  const ready = classifier !== null && modelStatus === 'ready';
  log.debug('model ready check', { ready });
  return ready;
}


/**
 * determines if an image is a meme using confidence-weighted classification.
 * uses both absolute confidence and relative margin to handle uncertain cases.
 *
 * algorithm:
 * 1. high confidence (winner > 0.8): trust the prediction
 * 2. low confidence (winner < 0.6): default to meme (uncertain cases)
 * 3. medium confidence (0.6-0.8): use margin analysis
 *    - small margin (< 0.3): lean toward meme
 *    - large margin (>= 0.3): trust the winner
 *
 * examples:
 * - meme: 0.63, not-meme: 0.37 (winner: 0.63, margin: 0.26) → meme = true (uncertain)
 * - meme: 0.08, not-meme: 0.92 (winner: 0.92, margin: 0.84) → meme = false (confident)
 * - meme: 0.82, not-meme: 0.18 (winner: 0.82, margin: 0.64) → meme = true (confident)
 * - meme: 0.45, not-meme: 0.55 (winner: 0.55, margin: 0.10) → meme = true (uncertain)
 * - meme: 0.25, not-meme: 0.75 (winner: 0.75, margin: 0.50) → meme = false (medium confidence, large margin)
 *
 */
function determineMemeClassification(
  results: Array<{ label: string; score: number }>
): { isMeme: boolean; confidence: number } {
  const memeResult = results.find(result =>
    result.label?.toLowerCase().includes('meme') &&
    !result.label?.toLowerCase().includes('not')
  );

  const notMemeResult = results.find(result =>
    result.label?.toLowerCase().includes('not meme') ||
    (result.label?.toLowerCase().includes('meme') && result.label?.toLowerCase().includes('not'))
  );

  const memeScore = memeResult?.score || 0.0;
  const notMemeScore = notMemeResult?.score || 0.0;

  // determine winner and margin
  const winnerScore = Math.max(memeScore, notMemeScore);
  const margin = Math.abs(memeScore - notMemeScore);
  const memeIsWinner = memeScore > notMemeScore;

  // confidence thresholds
  const highConfidenceThreshold = 0.8;
  const lowConfidenceThreshold = 0.6;
  const smallMarginThreshold = 0.3;

  let isMeme: boolean;
  if (winnerScore >= highConfidenceThreshold) {  // high confidence: trust the winner
    isMeme = memeIsWinner;
  } else if (winnerScore < lowConfidenceThreshold) {  // low confidence: default to meme (uncertain cases favor memes)
    isMeme = true;
  } else {  // medium confidence: use margin analysis
    if (margin < smallMarginThreshold) {
      isMeme = true;  // small margin: lean toward meme
    } else {
      isMeme = memeIsWinner;  // large margin: trust the winner
    }
  }

  return {
    isMeme,
    confidence: memeScore
  };
}
