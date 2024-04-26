storage "file" {
  path    = "./vault/data"
  node_id = "my_nodeid"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = "true"
}

api_addr = "http://0.0.0.0:8200"
cluster_addr = "https://0.0.0.0:8201"

disable_mlock = true
ui = true
