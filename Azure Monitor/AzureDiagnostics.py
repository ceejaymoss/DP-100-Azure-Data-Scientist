#
# This code is used in Azure monitor
AzureDiagnostics
    take 5
#
# Generates the latest 5
AzureDiagnostics
    top 5 by TimeGenerated
#
# Search for a specific piece of text
Search in (AzureDiagnostics) "Response"
    Top 5 by TimeGenerated
#
#
AzureDiagnostics
    where Resource =~ "Request"
    Top 5 by TimesGenerated
#
#
AzureDiagnostics
    where Resource =~ "Request"
    project TimeGenerated, Resource, status_s
#
#
AzureDiagnostics
    summarise count()
