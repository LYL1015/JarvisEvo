return {
    LrSdkVersion = 3.0,
    LrSdkMinimumVersion = 1.3,
    LrToolkitIdentifier = 'com.example.xmpplayer',
    LrPluginName = 'XMP Player',
    
    LrInitPlugin = 'InitPlugin.lua',  -- 指向初始化文件
    
    LrExportServiceProvider = {
        title = "XMP Player",
        file = 'ExportServiceProvider.lua',
    },

    VERSION = { major=1, minor=0, revision=0 },
} 