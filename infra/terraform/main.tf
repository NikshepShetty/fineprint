data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-rg"
  location = var.location
}

resource "azurerm_container_registry" "main" {
  name                = "${var.project_name}acr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
}

resource "azurerm_key_vault" "main" {
  name                = "${var.project_name}-kv"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  enable_rbac_authorization = true

  # Off deliberately - this is a demo vault meant to be destroyed and
  # recreated freely. Purge protection would block that for 90 days.
  purge_protection_enabled = false
}

resource "azurerm_role_assignment" "current_user_kv_admin" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id          = data.azurerm_client_config.current.object_id
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.project_name}-logs"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "main" {
  name                       = "${var.project_name}-env"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
}

resource "azurerm_container_app" "main" {
  name                         = "${var.project_name}-api"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"

  identity {
    type = "SystemAssigned"
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username             = azurerm_container_registry.main.admin_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.main.admin_password
  }

  secret {
    name                = "anthropic-api-key"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/ANTHROPIC-API-KEY"
    identity            = "System"
  }

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "fineprint-api"
      # placeholder - the CD workflow replaces this with the real built image
      image  = "mcr.microsoft.com/k8se/quickstart:latest"
      cpu    = 0.25
      memory = "0.5Gi"

      env {
        name        = "ANTHROPIC_API_KEY"
        secret_name = "anthropic-api-key"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].container[0].image,
    ]
  }

  depends_on = [azurerm_role_assignment.current_user_kv_admin]
}

resource "azurerm_role_assignment" "container_app_kv_reader" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_container_app.main.identity[0].principal_id
}
