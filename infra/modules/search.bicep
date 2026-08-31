@description('Azure AI Search service name — becomes the *.search.windows.net subdomain.')
param name string
param location string
param tags object

@description('Service SKU. basic/standard support vector + hybrid search.')
param sku string

resource search 'Microsoft.Search/searchServices@2024-06-01-preview' = {
  name: name
  location: location
  tags: tags
  sku: { name: sku }
  identity: { type: 'SystemAssigned' }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    // Enables the semantic reranker (free plan: 1,000 queries/month). Used as the
    // relevance gate + reranking in the query pipeline.
    semanticSearch: 'free'
    // Allow both API keys and Entra ID (RBAC). The local app uses the admin key;
    // this also lets you switch to managed-identity auth later without redeploying.
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
  }
}

output id string = search.id
output name string = search.name
output endpoint string = 'https://${search.name}.search.windows.net'
