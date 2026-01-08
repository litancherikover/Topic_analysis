# Guide for Large Parquet Files (Gigabytes)

## ✅ Yes, it can work! Here's how:

The app now supports **Parquet files** and includes optimizations for large datasets.

## Important Considerations

### 1. **GitHub File Size Limits**
- **Regular files**: 100MB limit
- **Git LFS (Large File Storage)**: Required for files >100MB
- **Recommended**: Use Git LFS for files >50MB

### 2. **Memory Requirements**
- Large files need sufficient RAM
- Use **chunking** for files >1GB
- Parquet is more memory-efficient than CSV

### 3. **Loading Speed**
- GitHub downloads can be slow for large files
- Consider using:
  - **Local files** (fastest)
  - **Cloud storage** (S3, Google Cloud Storage, etc.)
  - **CDN** for faster downloads

## How to Use Large Parquet Files

### Option 1: Upload via Streamlit UI (Recommended for Testing)
1. Run the Streamlit app
2. In sidebar, select **"Local File"**
3. Upload your `.parquet` file
4. Enable **"Use Chunking"** if file is >1GB
5. Adjust chunk size (default: 100,000 rows)

### Option 2: GitHub with Git LFS
1. **Install Git LFS**:
   ```bash
   git lfs install
   ```

2. **Track parquet files**:
   ```bash
   git lfs track "*.parquet"
   git add .gitattributes
   ```

3. **Add and commit your large file**:
   ```bash
   git add data.parquet
   git commit -m "Add large parquet file"
   git push
   ```

4. **In Streamlit app**:
   - Select "GitHub Parquet"
   - Enter the raw URL: `https://raw.githubusercontent.com/litancherikover/Topic_analysis/main/data.parquet`
   - Enable chunking if needed

### Option 3: Custom URL (Cloud Storage)
1. Upload your parquet file to:
   - AWS S3 (with public access)
   - Google Cloud Storage
   - Azure Blob Storage
   - Any public URL

2. In Streamlit app:
   - Select "Custom URL"
   - Enter the file URL
   - Enable chunking for large files

## Performance Tips

### For Files < 1GB
- ✅ No chunking needed
- ✅ Fast loading
- ✅ Works well from GitHub

### For Files 1-5GB
- ✅ Enable chunking
- ✅ Use chunk size: 100,000-500,000 rows
- ⚠️ Consider local file or cloud storage

### For Files > 5GB
- ✅ Use chunking (100,000-200,000 rows)
- ✅ Use local file or cloud storage (not GitHub)
- ✅ Consider pre-filtering data before loading
- ⚠️ May need more RAM

## Memory Optimization Features

The app includes:
- **Chunked loading**: Loads data in smaller pieces
- **Caching**: Streamlit caches loaded data
- **Efficient filtering**: Filters before aggregation
- **Parquet support**: More efficient than CSV

## Example: Loading a 2GB Parquet File

```python
# In Streamlit sidebar:
1. Select "Local File" or "Custom URL"
2. Enable "Use Chunking"
3. Set chunk size to 200,000
4. Upload/enter file path
5. Wait for loading (may take 1-2 minutes)
```

## Troubleshooting

### "Out of Memory" Error
- ✅ Enable chunking
- ✅ Reduce chunk size
- ✅ Use a machine with more RAM
- ✅ Pre-filter data before loading

### Slow Loading from GitHub
- ✅ Use local file instead
- ✅ Use cloud storage (S3, GCS)
- ✅ Enable chunking

### File Too Large for GitHub
- ✅ Use Git LFS
- ✅ Use cloud storage
- ✅ Split into smaller files

## Best Practices

1. **For production**: Use cloud storage (S3, GCS) instead of GitHub
2. **For development**: Use local files or smaller test datasets
3. **For sharing**: Use Git LFS for files >100MB
4. **For analysis**: Enable chunking for files >1GB

## Alternative: Pre-process Large Files

If files are too large, consider:
1. **Pre-aggregate** data before loading
2. **Filter** by date/category before uploading
3. **Sample** data for initial analysis
4. **Partition** by date or category into smaller files
