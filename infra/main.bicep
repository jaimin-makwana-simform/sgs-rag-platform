targetScope = 'resourceGroup'

@description('Short name used as a prefix for all resource names (e.g. "sgs-rag").')
param environmentName string

@description('Azure region for all resources.')
param location string = resourceGroup().location

// ---- Azure OpenAI models ----
// NOTE: gpt-4o-mini (2024-07-18) was deprecated 2026-03-31. gpt-5.1 is current and,
// crucially, available as regional "Standard" SKU (the mini/nano gpt-5.x variants are
// GlobalStandard-only, which this subscription's fraud guard currently blocks).
@description('Chat model to deploy (deployment name = model name).')
param chatModelName string = 'gpt-5.1'
@description('Chat model version.')
param chatModelVersion string = '2025-11-13'
@description('Chat deployment SKU name. Regional "Standard" avoids the Azure OpenAI "unusual activity" guard that can block GlobalStandard on some accounts.')
param chatSku string = 'Standard'
@description('Chat deployment capacity (TPM / 1000).')
param chatCapacity int = 20

@description('Embedding model to deploy (deployment name = model name).')
param embeddingModelName string = 'text-embedding-3-small'
@description('Embedding model version.')
param embeddingModelVersion string = '1'
@description('Embedding deployment SKU name.')
param embeddingSku string = 'Standard'
@description('Embedding deployment capacity (TPM / 1000).')
param embeddingCapacity int = 50

// ---- Azure AI Search ----
@description('Azure AI Search SKU. basic/standard support vector + hybrid search.')
@allowed(['free', 'basic', 'standard', 'standard2', 'standard3'])
param searchSku string = 'basic'

@description('Name of the search index the app creates/uses.')
param searchIndexName string = 'sgs-docs'

var prefix = toLower(environmentName)
// Search + OpenAI endpoints are global DNS names, so make them unique per RG.
var suffix = take(uniqueString(resourceGroup().id), 6)
var tags = { environment: environmentName, project: 'sgs-rag-poc' }

module openai 'modules/openai.bicep' = {
  name: 'openai'
  params: {
    name: '${prefix}-openai-${suffix}'
    location: location
    tags: tags
    chatModelName: chatModelName
    chatModelVersion: chatModelVersion
    chatSku: chatSku
    chatCapacity: chatCapacity
    embeddingModelName: embeddingModelName
    embeddingModelVersion: embeddingModelVersion
    embeddingSku: embeddingSku
    embeddingCapacity: embeddingCapacity
  }
}

module search 'modules/search.bicep' = {
  name: 'search'
  params: {
    name: '${prefix}-search-${suffix}'
    location: location
    tags: tags
    sku: searchSku
  }
}

// Non-secret outputs consumed by deploy.sh to build the local .env.
// Keys are fetched separately by the script (not emitted as deployment outputs).
output AZURE_OPENAI_ENDPOINT string = openai.outputs.endpoint
output AZURE_OPENAI_ACCOUNT_NAME string = openai.outputs.name
output AZURE_OPENAI_CHAT_DEPLOYMENT string = openai.outputs.chatDeploymentName
output AZURE_OPENAI_EMBEDDING_DEPLOYMENT string = openai.outputs.embeddingDeploymentName
output AZURE_SEARCH_ENDPOINT string = search.outputs.endpoint
output AZURE_SEARCH_SERVICE_NAME string = search.outputs.name
output AZURE_SEARCH_INDEX_NAME string = searchIndexName
