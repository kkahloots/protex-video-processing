# Protex AI - Scalability & Production Considerations

## Overview

This document outlines how the pipeline scales from demo (single video) to production (1000+ cameras, 20+ countries).

**Key Improvements at Scale**:
- **Stage 1**: Distributed processing (Dask/Ray), streaming architecture → 10x throughput
- **Stage 2**: Model optimization (TensorRT), edge deployment → 5x throughput per GPU, 80% bandwidth reduction
- **Stage 3**: Stateless microservice, adaptive thresholds → sub-second latency
- **Stage 4**: Smart sampling, QA dashboards → 50% QA time reduction
- **Stage 5**: Real-time dashboards, stakeholder reports → continuous monitoring

**Cost Impact**: $800/day infrastructure vs $1.5M/day manual annotation (without pre-tagging)

---

## Current Implementation vs Production Scale

This take-home demonstrates the **core pipeline logic** with production-ready structure. Here's how each stage would scale at Protex's real deployment (1000+ cameras, 20+ countries):

---

## 🎯 Stage 1: Preprocessing (Video → Frames)

### Current Implementation
- Single-threaded video processing
- Local file I/O
- Sequential frame extraction

### Production Improvements

**1. Distributed Processing**
- **Dask/Ray**: Parallelize across video chunks (e.g., 1-minute segments)
- **Per-camera workers**: Each camera stream processed independently
- **Cloud storage**: 
  - AWS: S3 for video/frames, EFS for shared processing
  - Azure: Blob Storage for video/frames, Azure Files for shared processing

**2. Streaming Architecture**
- Process live camera feeds instead of batch video files
- Use **GStreamer** or **FFmpeg** for hardware-accelerated decoding
- **Message queues**:
  - AWS: Kinesis Data Streams or SQS for frame distribution
  - Azure: Event Hubs or Service Bus for frame distribution

**3. Smart Sampling**
- **Motion detection** (OpenCV background subtraction) before FPS sampling
- **Scene change detection** to avoid redundant static periods
- **Adaptive FPS**: Increase sampling during detected activity

**Cost Impact**: 10x throughput improvement, 60% reduction in storage costs

---

## 🤖 Stage 2: Pre-tagging (Detection)

### Current Implementation
- PyTorch model on single GPU
- Batch size = 64
- FasterRCNN ResNet50 FPN

### Production Improvements

**1. Model Optimization**
- **ONNX/TensorRT export**: 3-5x inference speedup
- **Model quantization** (INT8): 2x speedup, 4x memory reduction
- **YOLOv8/YOLOv9**: Consider faster architectures for deployment

**2. Distributed Inference**
- **Inference serving**:
  - AWS: SageMaker multi-model endpoints or ECS with Triton
  - Azure: Azure ML managed endpoints or AKS with Triton
- **Batch aggregation**: Collect frames from multiple cameras, batch to 32-64
- **Auto-scaling**: 
  - AWS: ECS/EKS auto-scaling based on SQS queue depth
  - Azure: AKS HPA based on Service Bus queue depth

**3. Edge Deployment**
- **NVIDIA Jetson** at camera edge for local pre-filtering
- Only send "interesting" frames to cloud for full processing
- Reduces bandwidth by 80-90%

**Cost Impact**: 5x throughput per GPU, 70% bandwidth reduction with edge filtering

---

## 🧹 Stage 3: Cleanup

### Current Implementation
- Single-threaded COCO filtering
- In-memory processing

### Production Improvements

**1. Stateless Service**
- **FastAPI/Flask** microservice with configurable thresholds
- **Horizontal scaling**:
  - AWS: ECS Fargate or Lambda for serverless
  - Azure: Container Apps or Azure Functions for serverless
- **Cache**:
  - AWS: ElastiCache (Redis) for category lookups
  - Azure: Azure Cache for Redis for category lookups

**2. Adaptive Thresholds**
- Per-camera calibration (different MIN_AREA for different camera distances)
- **Time-of-day adjustments** (different thresholds for day/night)
- **Feedback loop**: Adjust thresholds based on annotator rejection rates

**3. Quality Metrics**
- Track false positive rates per camera
- **Anomaly detection**: Flag cameras with unusual detection patterns
- **Drift monitoring**: Alert when detection distributions change

**Cost Impact**: Sub-second latency, enables real-time annotation workflows

---

## 📊 Stage 4: Sample Generation

### Current Implementation
- Random sampling for QA
- Local image rendering

### Production Improvements

**1. Smart Sampling**
- **Stratified sampling**: Ensure coverage across classes, times, cameras
- **Hard example mining**: Prioritize ambiguous detections for review
- **Diversity sampling**: Maximize visual variety in sample set

**2. QA Dashboard Integration**
- **Web UI** for annotator review
- **A/B testing**: Compare different model versions side-by-side
- **Annotation metrics**: Track annotator agreement, time-per-image

**3. Continuous Monitoring**
- **Daily sample reports** per camera/site
- **Drift detection**: Alert when class distributions shift
- **Model performance tracking**: Precision/recall over time

**Cost Impact**: 50% reduction in annotation QA time, faster model iteration

---

## 📈 Stage 5: Report Generation

### Production Improvements

**1. Real-time Dashboards**
- **Grafana/Tableau**: Live metrics on annotation workload
- **Cost projections**: Estimate annotation cost based on current detection rates
- **SLA tracking**: Monitor pipeline latency end-to-end

**2. Stakeholder Reports**
- **Weekly summaries** for site managers
- **Monthly cost reports** for finance
- **Incident reports**: Automated summaries for safety events

---

## 🏗️ Overall Architecture at Scale

### Orchestration
- **Workflow engines**:
  - AWS: Step Functions, MWAA (Managed Airflow), or EventBridge
  - Azure: Logic Apps, Data Factory, or AKS with Argo Workflows
- **Event-driven**: 
  - AWS: S3 events → Lambda/Step Functions
  - Azure: Blob Storage events → Functions/Logic Apps
- **Retry logic**: Handle transient failures gracefully

### Data Management
- **Data versioning**: DVC/Pachyderm or cloud-native solutions
  - AWS: S3 versioning + Glue Data Catalog
  - Azure: Blob versioning + Purview Data Catalog
- **COCO dataset registry**: Centralized catalog of all annotation batches
- **Retention policies**: 
  - AWS: S3 Lifecycle policies (Glacier for archive)
  - Azure: Blob lifecycle management (Archive tier)

### Observability
- **Metrics & dashboards**:
  - AWS: CloudWatch + Managed Grafana
  - Azure: Azure Monitor + Managed Grafana
- **Error tracking**: Sentry/Datadog or cloud-native
  - AWS: CloudWatch Logs Insights + X-Ray
  - Azure: Application Insights + Log Analytics
- **Cost tracking**: 
  - AWS: Cost Explorer with resource tags per stage
  - Azure: Cost Management with resource tags per stage

### Security & Compliance
- **Encryption**:
  - AWS: S3 SSE-KMS, TLS in transit
  - Azure: Storage Service Encryption, TLS in transit
- **Access controls**: 
  - AWS: IAM roles + S3 bucket policies
  - Azure: Azure AD + RBAC + Storage access policies
- **Audit logs**: 
  - AWS: CloudTrail for all API calls
  - Azure: Activity Log for all operations
- **GDPR compliance**: Automated PII detection and redaction
  - AWS: Rekognition for face detection, Comprehend for PII
  - Azure: Computer Vision for face detection, Text Analytics for PII

---

## 💰 Cost Model at Scale

### Current Take-home (1 video)
- Processing time: ~5 minutes
- Storage: ~100 MB
- Annotation workload: ~50 images

### Production Scale (1000 cameras, 24/7)
- **Video ingestion**: 1000 cameras × 24 hours × 1 GB/hour = 24 TB/day
- **Frame extraction**: 1000 cameras × 86,400 sec/day × 1 FPS = 86M frames/day
- **After filtering**: ~30M frames/day (65% reduction)
- **Detection inference**: 30M frames / 100 frames/sec/GPU = 83 GPU-hours/day
- **Cloud costs** (approximate):
  - AWS: ~$200/day GPU (p3.2xlarge spot), ~$600/day S3 storage
  - Azure: ~$220/day GPU (NC6s v3 spot), ~$580/day Blob storage
- **Annotation workload**: 30M frames × $0.05/frame = $1.5M/day (if no pre-tagging)
- **With pre-tagging**: Reduce manual work by 70% → $450K/day

**Key insight**: Pre-tagging ROI is massive at scale. Even a 50% reduction in annotator effort saves $500K+/day.

---

---

## 🔧 Background Processing

For long-running pipelines, use background processing:

```bash
# Run pipeline in background
nohup runnable/run_pipeline.sh balanced data/video.mp4 > runnable/run_pipeline.log 2>&1 &

# Monitor progress
tail -f runnable/run_pipeline.log

# Check last run logs
cat runnable/run_pipeline.log
```

**For presentation guidance, see**: [docs/PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)
