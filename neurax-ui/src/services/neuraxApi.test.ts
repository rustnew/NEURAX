/**
 * NEURAX API Client Tests
 *
 * Tests the API client functions with mocked fetch responses.
 * Verifies that the client correctly constructs requests,
 * handles authentication, parses responses, and handles errors.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock supabase before importing neuraxApi
vi.mock('@/lib/supabaseClient', () => ({
  supabase: {
    auth: {
      getSession: vi.fn().mockResolvedValue({ data: { session: { access_token: 'test-token' } }, error: null }),
    },
  },
}));

// Mock import.meta.env
vi.stubGlobal('import.meta', {
  env: {
    VITE_NEURAX_API_URL: 'http://127.0.0.1:9098',
    VITE_SUPABASE_DISABLED: 'true',
  },
});

import {
  setNeuraxAccessToken,
  getHealth,
  listHardware,
  analyze,
  simulateInference,
  compareAnalyses,
  exportOnnx,
  listProjects,
  createProject,
  deleteProject,
  getCredits,
  getComplianceConfig,
} from './neuraxApi';

// Set a test token so we don't need real auth
setNeuraxAccessToken('test-token');

function mockFetch(response: unknown, status = 200) {
  return vi.spyOn(globalThis, 'fetch').mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(response),
  } as Response);
}

describe('neuraxApi', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setNeuraxAccessToken('test-token');
  });

  describe('getHealth', () => {
    it('should call GET /health', async () => {
      const fetchSpy = mockFetch({ status: 'ok' });

      const result = await getHealth();

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect((fetchSpy.mock.calls[0][0] as string)).toContain('/health');
      expect(result.status).toBe('ok');

      fetchSpy.mockRestore();
    });
  });

  describe('listHardware', () => {
    it('should call GET /hardware and return GPU list', async () => {
      const mockGpus = [
        { name: 'A100-SXM', manufacturer: 'NVIDIA', memory_gb: 80, memory_bandwidth_gbs: 2039, tflops_fp64: 19.5, tflops_fp32: 19.5, tflops_fp16: 312, tflops_bf16: 312, tflops_int8: 624, tflops_fp8: 1248, tensor_cores: true, nvlink: true, nvlink_bandwidth_gbs: 600, tdp_watts: 400, launch_year: 2020 },
        { name: 'H100-SXM', manufacturer: 'NVIDIA', memory_gb: 80, memory_bandwidth_gbs: 3350, tflops_fp64: 34, tflops_fp32: 67, tflops_fp16: 990, tflops_bf16: 990, tflops_int8: 1979, tflops_fp8: 3958, tensor_cores: true, nvlink: true, nvlink_bandwidth_gbs: 900, tdp_watts: 700, launch_year: 2022 },
      ];
      const fetchSpy = mockFetch(mockGpus);

      const result = await listHardware();

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect((fetchSpy.mock.calls[0][0] as string)).toContain('/hardware');
      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('A100-SXM');

      fetchSpy.mockRestore();
    });
  });

  describe('analyze', () => {
    it('should call POST /analyze with topology', async () => {
      const mockReport = { total_params: 124439808, forward_flops: 7.63e11 };
      const fetchSpy = mockFetch({ report: mockReport });

      const topology = { model: { name: 'test', type: 'transformer' } };
      const result = await analyze({ topology });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/analyze');
      expect(options.method).toBe('POST');
      expect(JSON.parse(options.body as string)).toEqual({ topology });

      fetchSpy.mockRestore();
    });
  });

  describe('simulateInference', () => {
    it('should call POST /inference/simulate', async () => {
      const mockReport = { latency_ms: 12.5, throughput_tokens_per_s: 80000 };
      const fetchSpy = mockFetch({ report: mockReport });

      const result = await simulateInference({
        topology: { model: { name: 'test' } },
        params: { batch_size: 1, seq_len: 128, precision: 'fp16', hardware: 'A100-SXM', gpu_count: 1 },
      });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/inference/simulate');
      expect(options.method).toBe('POST');

      fetchSpy.mockRestore();
    });
  });

  describe('compareAnalyses', () => {
    it('should call POST /analyze/compare with configs', async () => {
      const mockResults = {
        results: [
          { label: 'A100-SXM fp16 b64 g1', hardware: 'A100-SXM', report: {} },
        ],
      };
      const fetchSpy = mockFetch(mockResults);

      const result = await compareAnalyses({
        topology: { model: { name: 'test' } },
        configs: [{ hardware: 'A100-SXM', precision: 'fp16', batch_size: 64 }],
      });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/analyze/compare');
      expect(options.method).toBe('POST');

      fetchSpy.mockRestore();
    });
  });

  describe('exportOnnx', () => {
    it('should call POST /export/onnx', async () => {
      const mockResponse = {
        data: 'base64encodeddata',
        model_name: 'test-model',
        node_count: 42,
        initializer_count: 15,
        size_bytes: 12345,
      };
      const fetchSpy = mockFetch(mockResponse);

      const result = await exportOnnx({ topology: { model: { name: 'test' } } });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/export/onnx');
      expect(options.method).toBe('POST');
      expect(result.model_name).toBe('test-model');

      fetchSpy.mockRestore();
    });
  });

  describe('project CRUD', () => {
    it('should call GET /projects', async () => {
      const fetchSpy = mockFetch({ projects: [] });

      const result = await listProjects();

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect((fetchSpy.mock.calls[0][0] as string)).toContain('/projects');

      fetchSpy.mockRestore();
    });

    it('should call POST /projects to create', async () => {
      const mockProject = { id: 'uuid', name: 'Test Project' };
      const fetchSpy = mockFetch({ project: mockProject });

      await createProject({ name: 'Test Project', architecture: 'transformer' });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/projects');
      expect(options.method).toBe('POST');

      fetchSpy.mockRestore();
    });

    it('should call DELETE /projects/{id}', async () => {
      const fetchSpy = mockFetch({ deleted: true });

      await deleteProject('test-uuid');

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/projects/test-uuid');
      expect(options.method).toBe('DELETE');

      fetchSpy.mockRestore();
    });
  });

  describe('credits', () => {
    it('should call GET /credits', async () => {
      const mockCredits = { credits: { user_id: 'u1', used: 5, limit: 10, plan: 'free' } };
      const fetchSpy = mockFetch(mockCredits);

      const result = await getCredits();

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect((fetchSpy.mock.calls[0][0] as string)).toContain('/credits');
      expect(result.credits.plan).toBe('free');

      fetchSpy.mockRestore();
    });
  });

  describe('compliance', () => {
    it('should call GET /compliance/config', async () => {
      const mockCompliance = {
        regulations: [],
        thresholds: { high_risk_gflops: 10, carbon_report_tonnes: 1000 },
        recommendations: [],
      };
      const fetchSpy = mockFetch(mockCompliance);

      const result = await getComplianceConfig();

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect((fetchSpy.mock.calls[0][0] as string)).toContain('/compliance/config');

      fetchSpy.mockRestore();
    });
  });

  describe('error handling', () => {
    it('should throw on 401 errors', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ error: 'Unauthorized' }),
      } as Response);

      await expect(getHealth()).rejects.toThrow();

      fetchSpy.mockRestore();
    });

    it('should include Authorization header when token is set', async () => {
      const fetchSpy = mockFetch({ status: 'ok' });

      await getHealth();

      const options = fetchSpy.mock.calls[0][1] as RequestInit;
      expect(options.headers).toHaveProperty('Authorization', 'Bearer test-token');

      fetchSpy.mockRestore();
    });
  });
});