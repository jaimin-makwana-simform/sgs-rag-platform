@description('Azure OpenAI (Cognitive Services) account name — becomes the *.openai.azure.com subdomain.')
param name string
param location string
param tags object

param chatModelName string
param chatModelVersion string
param chatSku string
param chatCapacity int

param embeddingModelName string
param embeddingModelVersion string
param embeddingSku string
param embeddingCapacity int

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: { name: 'S0' }
  identity: { type: 'SystemAssigned' }
  properties: {
    // Required so the account gets a *.openai.azure.com endpoint and supports
    // Entra ID token auth. Key-based auth stays enabled for the local app.
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: account
  name: embeddingModelName
  sku: {
    name: embeddingSku
    capacity: embeddingCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: embeddingModelName
      version: embeddingModelVersion
    }
  }
}

// Deploy the chat model AFTER the embedding one — Cognitive Services rejects
// concurrent deployment operations on the same account.
resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: account
  name: chatModelName
  sku: {
    name: chatSku
    capacity: chatCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: chatModelName
      version: chatModelVersion
    }
  }
  dependsOn: [embeddingDeployment]
}

output id string = account.id
output name string = account.name
output endpoint string = account.properties.endpoint
output chatDeploymentName string = chatDeployment.name
output embeddingDeploymentName string = embeddingDeployment.name
