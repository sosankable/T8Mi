# Computer Vision Line Chat Bot

## On Azure

### Resource

- Azure App Service: B1

### App service setting

- TLS/SSL setting: 
    - HTTPS only: on 

### Prepare 

1. Install azure cli

2. Turn on Azure app service

3. Open SSH session in browser

4. Edit `/home/config.json`
```
{
    "line":{
            "line_secret":...,
            "line_token":...
    },
    "azure":{
            "subscription_key":...,
            "endpoint":"https://<your name of Azure Cognitive Services>.cognitiveservices.azure.com/",
            "face_key":...,
            "face_end":"https://<your name of Azure Face Detection>.cognitiveservices.azure.com/"
    },
    "imgur":{
            "client_id":...,
            "client_secret":...,
            "access_token":...,
            "refresh_token":...
    }
}
```
5. Set username and password: `az webapp deployment user set --user-name <usrname> --password <password>`

6. Get git url:
`az webapp deployment source config-local-git --name <app_name> --resource-group <resource_name>`

7. Add remote: 
```
cd cv_tutorial
git remote add azure <your_git_url>
```

8. `git push azure master`

