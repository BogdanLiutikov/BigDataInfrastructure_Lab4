#!/usr/bin/env sh

# Start vault
dumb-init vault server -dev &

# Wait for vault to be ready
until vault status > /dev/null 2>&1; do
    echo wait
    sleep 3
done
echo "Vault is ready."

vault login token=$VAULT_DEV_ROOT_TOKEN_ID
# vault secrets enable -version=2 -path=secrets kv
vault kv put secret/db MSSQL_USER=$MSSQL_USER MSSQL_SA_PASSWORD=$MSSQL_SA_PASSWORD
tail -f /dev/null