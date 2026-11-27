import { Router } from "express";

import { GraphEstimationError } from "../graphEstimationClient.js";
import { UnavailableError } from "../estimatorClient.js";

const MIN_TRANSCRIPT = 100;

export function graphEstimationsRouter({ graphClient }) {
  const router = Router();

  router.get("/graph-estimations/new", (req, res) => {
    res.render("graph-estimations/new", { values: {}, error: "" });
  });

  router.post("/graph-estimations", async (req, res, next) => {
    const transcript = String(req.body.transcript || "").trim();
    if (transcript.length < MIN_TRANSCRIPT) {
      res.status(422).render("graph-estimations/new", {
        values: { transcript },
        error: `La transcripción debe tener al menos ${MIN_TRANSCRIPT} caracteres.`,
      });
      return;
    }

    try {
      const result = await graphClient.estimateFromTranscript(transcript);
      res.render("graph-estimations/show", { result, transcript, error: "" });
    } catch (error) {
      if (error instanceof GraphEstimationError || error instanceof UnavailableError) {
        const status = error.status === 502 || error.status === 503 ? error.status : 400;
        res.status(status).render("graph-estimations/new", {
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
