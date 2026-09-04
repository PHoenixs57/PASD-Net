#!/bin/sh
set -e

hash=`cat model_version`
model=pasdnet_data-$hash.tar.gz

# This release does not bundle a model download source. Publish the model
# archive to your own GitHub Release / LFS / object storage and provide a
# direct download URL through the PASDNET_MODEL_URL environment variable.
if [ -z "${PASDNET_MODEL_URL}" ]; then
   echo "PASDNET_MODEL_URL is not set."
   echo "Please set PASDNET_MODEL_URL to a direct download URL for: ${model}"
   echo "Example: export PASDNET_MODEL_URL=https://.../${model}"
   exit 1
fi

if [ ! -f $model ]; then
   echo "Downloading model: ${model}"
   wget -O "$model" "$PASDNET_MODEL_URL"
fi

if command -v sha256sum
then
   echo "Validating checksum"
   checksum="$hash"
   checksum2=$(sha256sum $model | awk '{print $1}')
   if [ "$checksum" != "$checksum2" ]
   then
      echo "Aborting due to mismatching checksums. This could be caused by a corrupted download of $model."
      echo "Consider deleting local copy of $model and running this script again."
      exit 1
   else
      echo "checksums match"
   fi
else
   echo "Could not find sha256sum; skipping verification. Please verify manually that the sha256 hash of ${model} matches ${checksum}."
fi


tar xvomf $model

