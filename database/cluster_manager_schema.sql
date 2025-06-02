-- Cluster Manager Database Schema
-- PostgreSQL schema for cluster state and resource management

-- Cluster nodes tracking
CREATE TABLE IF NOT EXISTS cluster_nodes (
    id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    ip_address INET NOT NULL,
    port INTEGER NOT NULL,
    node_type TEXT NOT NULL DEFAULT 'worker', -- 'master', 'worker', 'gateway'
    status TEXT NOT NULL DEFAULT 'unknown', -- 'online', 'offline', 'maintenance', 'unknown'
    last_heartbeat TIMESTAMP,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    left_at TIMESTAMP,
    version TEXT,
    capabilities JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- GPU resource tracking
CREATE TABLE IF NOT EXISTS gpu_resources (
    id SERIAL PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
    gpu_index INTEGER NOT NULL,
    gpu_name TEXT,
    memory_total_mb INTEGER,
    memory_used_mb INTEGER DEFAULT 0,
    memory_available_mb INTEGER,
    utilization_percent REAL DEFAULT 0.0,
    temperature_celsius REAL,
    power_usage_watts REAL,
    driver_version TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(node_id, gpu_index)
);

-- System resource tracking
CREATE TABLE IF NOT EXISTS system_resources (
    id SERIAL PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
    cpu_count INTEGER,
    cpu_usage_percent REAL DEFAULT 0.0,
    memory_total_mb INTEGER,
    memory_used_mb INTEGER DEFAULT 0.0,
    memory_available_mb INTEGER,
    disk_total_gb INTEGER,
    disk_used_gb INTEGER DEFAULT 0.0,
    disk_available_gb INTEGER,
    network_rx_bytes BIGINT DEFAULT 0,
    network_tx_bytes BIGINT DEFAULT 0,
    load_average_1m REAL,
    load_average_5m REAL,
    load_average_15m REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(node_id)
);

-- Resource allocation tracking
CREATE TABLE IF NOT EXISTS resource_allocations (
    id SERIAL PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL, -- 'gpu', 'cpu', 'memory'
    resource_id TEXT, -- GPU index, CPU core range, etc.
    allocated_to TEXT, -- worker_id, task_id, service_name
    allocation_type TEXT NOT NULL, -- 'worker', 'task', 'service'
    allocated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    released_at TIMESTAMP,
    allocation_size INTEGER, -- MB for memory, % for CPU/GPU
    priority INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Cluster events and alerts
CREATE TABLE IF NOT EXISTS cluster_events (
    id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL, -- 'node_join', 'node_leave', 'resource_alert', 'health_check'
    severity TEXT NOT NULL DEFAULT 'info', -- 'info', 'warning', 'error', 'critical'
    source_node_id TEXT REFERENCES cluster_nodes(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    description TEXT,
    event_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    acknowledged_by TEXT,
    resolved_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Load balancing metrics
CREATE TABLE IF NOT EXISTS load_balancing_metrics (
    id SERIAL PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_unit TEXT,
    weight REAL DEFAULT 1.0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Cluster configuration
CREATE TABLE IF NOT EXISTS cluster_config (
    id SERIAL PRIMARY KEY,
    config_key TEXT NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    config_type TEXT DEFAULT 'string', -- 'string', 'integer', 'float', 'boolean', 'json'
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Network topology tracking
CREATE TABLE IF NOT EXISTS network_topology (
    id SERIAL PRIMARY KEY,
    source_node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
    connection_type TEXT DEFAULT 'network', -- 'network', 'direct', 'vpn'
    latency_ms REAL,
    bandwidth_mbps REAL,
    packet_loss_percent REAL DEFAULT 0.0,
    last_measured TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(source_node_id, target_node_id)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_cluster_nodes_status ON cluster_nodes(status);
CREATE INDEX IF NOT EXISTS idx_cluster_nodes_type ON cluster_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_cluster_nodes_heartbeat ON cluster_nodes(last_heartbeat);

CREATE INDEX IF NOT EXISTS idx_gpu_resources_node ON gpu_resources(node_id);
CREATE INDEX IF NOT EXISTS idx_gpu_resources_updated ON gpu_resources(last_updated);
CREATE INDEX IF NOT EXISTS idx_gpu_resources_utilization ON gpu_resources(utilization_percent);

CREATE INDEX IF NOT EXISTS idx_system_resources_node ON system_resources(node_id);
CREATE INDEX IF NOT EXISTS idx_system_resources_updated ON system_resources(last_updated);

CREATE INDEX IF NOT EXISTS idx_resource_allocations_node ON resource_allocations(node_id);
CREATE INDEX IF NOT EXISTS idx_resource_allocations_type ON resource_allocations(allocation_type);
CREATE INDEX IF NOT EXISTS idx_resource_allocations_allocated_to ON resource_allocations(allocated_to);
CREATE INDEX IF NOT EXISTS idx_resource_allocations_active ON resource_allocations(allocated_at) WHERE released_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_cluster_events_type ON cluster_events(event_type);
CREATE INDEX IF NOT EXISTS idx_cluster_events_severity ON cluster_events(severity);
CREATE INDEX IF NOT EXISTS idx_cluster_events_created_at ON cluster_events(created_at);
CREATE INDEX IF NOT EXISTS idx_cluster_events_unresolved ON cluster_events(created_at) WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_load_balancing_node ON load_balancing_metrics(node_id);
CREATE INDEX IF NOT EXISTS idx_load_balancing_metric ON load_balancing_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_load_balancing_recorded_at ON load_balancing_metrics(recorded_at);

CREATE INDEX IF NOT EXISTS idx_network_topology_source ON network_topology(source_node_id);
CREATE INDEX IF NOT EXISTS idx_network_topology_target ON network_topology(target_node_id);

-- JSONB indexes for metadata searches
CREATE INDEX IF NOT EXISTS idx_cluster_nodes_metadata ON cluster_nodes USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_cluster_nodes_capabilities ON cluster_nodes USING GIN(capabilities);
CREATE INDEX IF NOT EXISTS idx_cluster_events_data ON cluster_events USING GIN(event_data);

-- Cluster health view
CREATE OR REPLACE VIEW cluster_health_summary AS
SELECT 
    COUNT(*) as total_nodes,
    COUNT(*) FILTER (WHERE status = 'online') as online_nodes,
    COUNT(*) FILTER (WHERE status = 'offline') as offline_nodes,
    COUNT(*) FILTER (WHERE status = 'maintenance') as maintenance_nodes,
    COUNT(*) FILTER (WHERE last_heartbeat > NOW() - INTERVAL '5 minutes') as healthy_nodes,
    (SELECT COUNT(*) FROM gpu_resources WHERE last_updated > NOW() - INTERVAL '5 minutes') as active_gpus,
    (SELECT SUM(memory_total_mb) FROM gpu_resources) as total_gpu_memory_mb,
    (SELECT SUM(memory_used_mb) FROM gpu_resources) as used_gpu_memory_mb,
    (SELECT AVG(utilization_percent) FROM gpu_resources WHERE last_updated > NOW() - INTERVAL '5 minutes') as avg_gpu_utilization
FROM cluster_nodes;

-- Resource utilization view
CREATE OR REPLACE VIEW resource_utilization AS
SELECT 
    cn.id as node_id,
    cn.hostname,
    cn.status as node_status,
    sr.cpu_usage_percent,
    sr.memory_used_mb::REAL / sr.memory_total_mb * 100 as memory_usage_percent,
    sr.disk_used_gb::REAL / sr.disk_total_gb * 100 as disk_usage_percent,
    COALESCE(gpu_stats.avg_gpu_utilization, 0) as avg_gpu_utilization,
    COALESCE(gpu_stats.gpu_count, 0) as gpu_count,
    sr.last_updated
FROM cluster_nodes cn
LEFT JOIN system_resources sr ON cn.id = sr.node_id
LEFT JOIN (
    SELECT 
        node_id,
        COUNT(*) as gpu_count,
        AVG(utilization_percent) as avg_gpu_utilization
    FROM gpu_resources 
    GROUP BY node_id
) gpu_stats ON cn.id = gpu_stats.node_id;

-- Active alerts view
CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    event_type,
    severity,
    COUNT(*) as alert_count,
    MAX(created_at) as latest_alert,
    MIN(created_at) as earliest_alert,
    array_agg(DISTINCT source_node_id) as affected_nodes
FROM cluster_events
WHERE resolved_at IS NULL 
  AND severity IN ('warning', 'error', 'critical')
  AND created_at > NOW() - INTERVAL '24 hours'
GROUP BY event_type, severity
ORDER BY 
    CASE severity 
        WHEN 'critical' THEN 1 
        WHEN 'error' THEN 2 
        WHEN 'warning' THEN 3 
    END,
    alert_count DESC;
