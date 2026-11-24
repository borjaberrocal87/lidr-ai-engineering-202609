import { Router } from "express";

import { AgentEstimationError } from "../agentEstimationClient.js";
import { UnavailableError } from "../estimatorClient.js";

const MIN_TRANSCRIPT = 100;

export function agentEstimationsRouter({ agentClient }) {
  const router = Router();

  router.get("/agent-estimations/new", (req, res) => {
    res.render("agent-estimations/new", { values: {}, error: "" });
  });

  router.post("/agent-estimations", async (req, res, next) => {
    const transcript = String(req.body.transcript || "").trim();
    if (transcript.length < MIN_TRANSCRIPT) {
      res.status(422).render("agent-estimations/new", {
        values: { transcript },
        error: `La transcripción debe tener al menos ${MIN_TRANSCRIPT} caracteres.`,
      });
      return;
    }

    try {
      const result = await agentClient.estimateFromTranscript(transcript);
      res.render("agent-estimations/show", { result, transcript, error: "" });
    } catch (error) {
      if (error instanceof AgentEstimationError || error instanceof UnavailableError) {
        const status = error.status === 502 ? 502 : 400;
        res.status(status).render("agent-estimations/new", {
          values: { transcript },
          error: error.message,
        });
        return;
      }
      next(error);
    }
  });

  return router;
}
