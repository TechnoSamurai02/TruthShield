import type { AnalysisResult } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function parseResponse(response: Response): Promise<AnalysisResult> {
  if (!response.ok) {
    let message = "We could not analyze this file. Try another file type or smaller file.";
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Keep the friendly default message.
    }
    throw new Error(message);
  }
  return response.json();
}

async function analyzeFile(endpoint: string, file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      body: formData
    });
    return parseResponse(response);
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("TruthShield could not reach the analysis service. Check the connection and try again.");
    }
    throw error;
  }
}

export function analyzeImage(file: File): Promise<AnalysisResult> {
  return analyzeFile("/api/analyze/image", file);
}

export function analyzeVideo(file: File): Promise<AnalysisResult> {
  return analyzeFile("/api/analyze/video", file);
}
