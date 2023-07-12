
        DECLARE @AP_MODULE_COMPONENT_ID int
        SELECT @AP_MODULE_COMPONENT_ID = AP_MODULE_COMPONENT_ID
        FROM [MIDDLEOFFICE_ALM].[dbo].[AP_MODULE_COMPONENT]
        WHERE AP_MODULE_CODE = 'ALLO'
        AND AP_COMPONENT_CODE = 'SMAA'

        INSERT INTO [MIDDLEOFFICE_ALM].[dbo].[AP_MODULE_COMPONENT_SNAPSHOT]
                   ([AP_MODULE_COMPONENT_ID]
                   ,[AS_OF_DATE]
                   ,[SNAPSHOT_VALUE]
                   ,[CREATED_DATE]
                   ,[MODIFIED_DATE]
                   ,[CREATED_BY]
                   ,[MODIFIED_BY]
                   ,[IS_ACTIVE])
             VALUES
                   (@AP_MODULE_COMPONENT_ID
                   ,GETDATE()
                   ,GETDATE()
                   ,GETDATE()
                   ,'System'
                   ,'System'
                   ,1)
    