using System;
using System.Threading.Tasks;
using Azure.Storage.Blobs;
using System.IO;

namespace MyApp
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            var connectionstring = "REPLACE ME";

            var container = new BlobContainerClient(connectionstring, "demo");
            await container.CreateIfNotExistsAsync();

            var filename = "demo.txt";
            var blob = container.GetBlobClient(filename);
            await blob.UploadAsync(Path.Combine("data", filename), true);

            await blob.DownloadToAsync(Path.Combine("Downloads", filename)); 

        }
    }
}