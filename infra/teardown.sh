#!/bin/bash
set -euo pipefail

# Deletes ONLY this project's resources from the (shared) resource group.
# It targets resources tagged `project=sgs-rag-poc` — the tag deploy.sh/Bicep
# stamps on everything it creates — so nothing else in the group is affected.
# There is intentionally NO "delete the whole resource group" option, because
# the group is shared.
#
#   RESOURCE_GROUP=AI-CoE-rg ./infra/teardown.sh

SUBSCRIPTION="${SUBSCRIPTION:-}"
RESOURCE_GROUP="${RESOURCE_GROUP:-AI-CoE-rg}"
PROJECT_TAG="${PROJECT_TAG:-sgs-rag-poc}"

if [[ -n "$SUBSCRIPTION" ]]; then
  az account set --subscription "$SUBSCRIPTION"
fi
echo ">>> Subscription    : $(az account show --query name -o tsv)"
echo ">>> Resource group  : $RESOURCE_GROUP"
echo ">>> Deleting only resources tagged: project=$PROJECT_TAG"

# Authoritative list: resources IN THIS RG carrying our project tag.
mapfile -t ROWS < <(az resource list \
  --resource-group "$RESOURCE_GROUP" \
  --query "[?tags.project=='${PROJECT_TAG}'].{name:name,type:type,location:location,id:id}" \
  -o tsv)

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo ">>> No resources tagged project=$PROJECT_TAG found in '$RESOURCE_GROUP'. Nothing to do."
  exit 0
fi

echo ""
echo ">>> The following resources will be DELETED:"
for row in "${ROWS[@]}"; do
  name="$(cut -f1 <<<"$row")"; rtype="$(cut -f2 <<<"$row")"
  echo "    - $name  ($rtype)"
done
echo ""
read -rp ">>> Proceed? [y/N] " c
[[ "${c:-N}" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

for row in "${ROWS[@]}"; do
  name="$(cut -f1 <<<"$row")"
  rtype="$(cut -f2 <<<"$row")"
  location="$(cut -f3 <<<"$row")"
  id="$(cut -f4 <<<"$row")"
  echo ">>> Deleting $name ($rtype)..."
  case "$rtype" in
    Microsoft.CognitiveServices/accounts)
      az cognitiveservices account delete -n "$name" -g "$RESOURCE_GROUP" -o none
      # Cognitive Services accounts are soft-deleted; purge so the name is reusable.
      az cognitiveservices account purge -n "$name" -g "$RESOURCE_GROUP" -l "$location" -o none 2>/dev/null \
        && echo "    purged soft-delete." || true
      ;;
    Microsoft.Search/searchServices)
      az search service delete -n "$name" -g "$RESOURCE_GROUP" --yes -o none
      ;;
    *)
      az resource delete --ids "$id" -o none
      ;;
  esac
  echo "    done."
done

echo ">>> Teardown complete — only project=$PROJECT_TAG resources were removed."
