import unreal

def importAssets(file_path):
    print(f"Importing asset from: {file_path}")

    # create asset tools object
    assetTools = unreal.AssetToolsHelpers.get_asset_tools()
    # create asset import data object        
    assetImportData = unreal.AutomatedAssetImportData()
    # set assetImportData attributes
    assetImportData.destination_path = '/Game/Python/GeneratedAssets'
    assetImportData.filenames = [file_path]
    assetImportData.replace_existing = True
    assetTools.import_assets_automated(assetImportData)

# example usage
# importAssets('D:/NAMIZJE/gorila.png')
# print("Asset import completed.")