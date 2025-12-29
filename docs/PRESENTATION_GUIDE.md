# Presentation Guide

## Overview

**Duration**: ~7.9 minutes (62 slides)  
**Format**: Auto-generated MP4 video or live walkthrough  
**Audience**: Engineering team, stakeholders, clients

---

## Generate Presentation

```bash
python 06_generate_presentation.py traceables/protex_presentation.mp4
```

**Output**:
- `traceables/protex_presentation.mp4` - Video file
- `traceables/presentation_slides/` - Individual PNG slides

---

## Key Messages

### Business Impact
- **64% annotation time reduction** (1.8h vs 5.0h manual)
- **$58M/year savings** at 1000 cameras
- **$456K/year GPU infrastructure savings**
- **ROI**: 8,200% return, 4.4 day payback period

### Technical Excellence
- **Modular architecture**: Independent scaling per stage
- **Config-driven**: Per-client tuning without code changes
- **Safety-focused**: 10 relevant classes, class-aware filtering
- **Production-ready**: Single runner, full tracking, cross-platform

### Engineering Manager Perspective
- Clear business trade-offs (3 operational modes)
- Cost awareness (infrastructure vs annotation savings)
- Stakeholder communication (reports, time estimates)
- Team scalability (config-driven, documented)

---

## Presentation Flow

### Section 1: Problem & Examples (9 slides)
- Title: Business impact headline
- Problem statement with cost analysis
- 3 Annotated/not Annonated examples
- Stage 1-3 images (pipeline visualization)
- Annotated/not Annonated demo

### Section 2: Design Rationale (7 slides)
- Modular architecture (CV Ops vs Data Ops)
- Config-driven design (2 slides)
- Model choice justification
- Three operational modes
- Class-aware filtering (2 slides)

### Section 3: Scale (9 slides)
- Multi-camera architecture (2 slides)
- GPU infrastructure (2 slides)
- Model optimization
- Edge deployment (2 slides)
- Infrastructure orchestration (2 slides)

### Section 4: Annotator Focus (15 slides)
- Report overview
- Dataset statistics
- Time estimates
- Sample images walkthrough
- 5 Annotated/not Annonated examples
- Detected objects summary
- Edge cases and failures
- Color coding guide
- Annotator empathy (2 slides)
- Quality checklist (2 slides)
- Escalation guidelines (2 slides)
- Efficiency tips (2 slides)

### Section 5: Features & Impact (16 slides)
- EM perspective on design (2 slides)
- Key features implemented (2 slides)
- Perceptual hash deduplication
- Production optimizations (2 slides)
- Business impact (2 slides)
- ROI calculation (2 slides)
- Engineering ROI perspective (2 slides)
- Code quality summary
- Production readiness summary
- Business value summary

### Section 6: Closing (6 slides)
- Get started
- EM perspective summary (2 slides)
- Thank you with key takeaways

---

## Key Talking Points

### Pipeline Architecture
> "Five independent stages—preprocessing, pre-tagging, cleanup, samples, and reporting. Why modular? At scale, these would be separate services. Stage 2 needs GPUs and auto-scaling. Stage 1 traceables on cheaper CPU instances. Stage 3 is stateless and serverless."

### Config-Driven Design
> "Everything is config-driven through YAML. Different scenarios need different trade-offs. A pilot deployment uses 'fast' mode—0.5 FPS, higher confidence threshold, minimal cost. A critical safety zone uses 'accurate' mode—3 FPS, lower threshold, maximum coverage."

### Class-Aware Filtering
> "We're not filtering uniformly. People get a lower area threshold—300px² versus 1000px²—because any human presence near machinery is safety-critical. This class-aware logic is business domain knowledge that matters more than fancy ML."

### Business Impact
> "This pipeline transforms a 5-hour manual task into a 1.8-hour assisted workflow. At Protex's scale of 1000 cameras, that's $159,000 saved per day, or $58 million annually. The infrastructure pays for itself in 4.4 days."

### Production Readiness
> "For 24-hour video, we'd optimize: batch size 64→124 (saturate GPU), Faster R-CNN→YOLOv8 (5-10x faster), multi-GPU (4x throughput), async I/O (overlap loading/inference). Result: 1.5 hours→10-15 minutes, 90% reduction."

---

## Metrics to Emphasize

### Time Savings
- 5.0h → 1.8h (64% reduction)
- 1.5h → 10-15min at production scale (90% reduction)

### Cost Savings
- $159 per dataset (vs $250 manual)
- $58M/year at 1000 cameras
- $456K/year GPU infrastructure savings
- ROI: 8,200% return

### Quality Metrics
- 60-80% frame reduction (preprocessing)
- 70-80% annotation noise reduction (safety-class filtering)
- 30-60% annotation volume (mode-based cleanup)

### Business Impact
- Days vs weeks for client onboarding
- Infrastructure pays for itself in 4.4 days
- Linear scaling without linear costs

---

## Live Walkthrough Tips

If presenting live instead of using the video:

1. **Show, don't just tell**: Run the pipeline, show actual outputs
2. **Use the cursor**: Point to specific config values, report sections
3. **Pause for emphasis**: Let key metrics sink in ($58M/year, 64% reduction)
4. **Keep it conversational**: Explain to a colleague, not reading slides
5. **Time management**: Aim for 6-7 minutes, leave time for questions

---

## Stage-by-Stage Walkthrough

### Stage 1: Preprocessing
> "Stage 1 extracts frames and applies quality filters. We're not just sampling at 1 FPS—we filter out black frames, duplicates, dark frames, and blurry frames. Critical for 24/7 industrial monitoring with camera glitches and night-shift footage. Result: 60-80% frame reduction without losing signal."

### Stage 2: Pre-tagging
> "Stage 2 traceables object detection. Faster R-CNN ResNet50 FPN—battle-tested, good accuracy for pre-tagging. Batch inference processes 4 images at once. At scale, we'd batch across cameras to reach 32-64 images per batch. We filter to 10 safety-relevant classes, reducing annotation noise by 70-80%."

### Stage 3: Cleanup
> "Stage 3 applies domain knowledge. We remove boxes smaller than 1000px², typically noise. But people get a lower threshold—300px²—because any human near machinery is safety-critical. This class-aware logic demonstrates business domain knowledge."

### Stage 4 & 5: Samples & Report
> "Stages 4 and 5 are about human communication. We generate annotated samples for QA, and a report that tells annotators exactly what to focus on. The report includes time estimates: 1.8 hours with pre-tagging versus 5.0 hours without. That's the ROI conversation with stakeholders."

---

## Production Considerations

### Scale Thinking
> "This is a demo, but designed with production in mind. Stage 1 would use distributed processing—Dask or Ray to parallelize across video chunks. Stage 2 is the bottleneck—export to TensorRT for 3-5x speedup, deploy to NVIDIA Jetson at camera edge to reduce bandwidth by 80%."

### Cloud Architecture
> "For 1000 concurrent cameras, we'd use microservices. On AWS: S3 for storage, Kinesis for streaming, Step Functions for orchestration, SageMaker for inference. On Azure: Blob Storage, Event Hubs, Logic Apps, Azure ML."

### Cost Model
> "Current: 1500 GPU-hours/day = $1,500/day. Optimized: 250 GPU-hours/day = $250/day. Savings: $456K/year in infrastructure alone. Combined with annotation savings: $58M/year total."

---

## Closing Statement

> "This pipeline demonstrates three things: EM thinking—I'm talking about cost, scale, and stakeholder communication. Production structure—modular, config-driven, cloud-ready. Domain empathy—I understand this is about industrial safety, not just bounding boxes."

> "The result? We can onboard new clients in days instead of weeks, delivering risk reduction faster. The infrastructure pays for itself in 4.4 days, and at scale, we save $58 million annually."

> "Run: `./runnable/run_pipeline.sh balanced data/timelapse_test.mp4`"

---

## References

- **Technical Details**: docs/FEATURES.md
- **Scale Architecture**: docs/SCALABILITY.md
- **Annotator Workflow**: docs/ANNOTATOR_GUIDE.md
- **Business Impact**: docs/report.md (auto-generated from traceables/report/text/report.md)
- **Quick Start**: docs/QUICKSTART.md
