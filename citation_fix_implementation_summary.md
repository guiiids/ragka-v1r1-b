# Citation Link Fix Implementation Summary

## Problem

The RAG assistant has an issue where inline citations in generated answers show a certain number of sources (e.g., 4), but the displayed source list only shows fewer sources (e.g., 3). Clicking on some inline citations results in "Source information not found." This causes confusion and breaks the citation functionality.

## Root Cause

The root cause of the issue is that the backend limits the number of parent documents to 3, which can exclude some cited sources. Additionally, the citation IDs are sequential numbers (1, 2, 3) that are regenerated for each query, causing inconsistency when citations are referenced across conversation turns.

## Solution

The solution implements a persistent citation ID system that maintains consistency across conversation turns:

1. **Unique Source ID Generation**: Each source now gets a unique, persistent ID in the format `S_{timestamp}_{hash}` instead of sequential numbers. This ensures that sources can be uniquely identified across conversation turns.

2. **Dual ID System**: The system now maintains both unique IDs (for internal tracking) and display IDs (for user-facing citations):
   - Unique IDs: Used internally to track sources across conversation turns
   - Display IDs: Sequential numbers (1, 2, 3) shown to users in the answer text

3. **Enhanced Citation Detection**: The `_filter_cited` method now detects both numeric citations and unique ID citations, ensuring backward compatibility.

4. **Cumulative Source Map**: The system maintains a cumulative source map across conversation turns, ensuring that all sources ever referenced are available for citation.

5. **Frontend Enhancements**: The frontend now handles both unique IDs and display IDs, ensuring that citation links work correctly.

## Implementation Details

### Backend Changes

1. **Unique ID Generation**:
   ```python
   def generate_unique_source_id(content: str = "", timestamp: float = None) -> str:
       """Generate a unique, persistent ID for a source."""
       if timestamp is None:
           timestamp = int(time.time() * 1000)
       
       hash_input = f"{content}_{timestamp}".encode('utf-8')
       content_hash = hashlib.md5(hash_input).hexdigest()[:8]
       
       return f"S_{timestamp}_{content_hash}"
   ```

2. **Context Preparation with Unique IDs**:
   ```python
   def _prepare_context(self, results: List[Dict]) -> Tuple[str, Dict]:
       # ...existing code...
       
       for res in prioritized_results[:5]:
           # ...existing code...
           
           # Generate unique ID for this source
           unique_id = generate_unique_source_id(chunk)
           
           # Include metadata in the source tag with unique ID
           entries.append(f'<source id="{unique_id}"{metadata_str}>{formatted_chunk}</source>')
           
           src_map[unique_id] = {
               "title": res["title"],
               "content": formatted_chunk,
               "parent_id": parent_id,
               "is_procedural": is_proc,
               "metadata": metadata
           }
       
       # ...rest of the method...
   ```

3. **Enhanced Citation Detection**:
   ```python
   def _filter_cited(self, answer: str, src_map: Dict) -> List[Dict]:
       # ...existing code...
       
       # Pattern for numeric citations [1], [2], etc.
       numeric_citation_pattern = r'\[(\d+)\]'
       for match in re.finditer(numeric_citation_pattern, answer):
           sid = match.group(1)
           if sid in src_map:
               explicit_citations.add(sid)
       
       # Pattern for unique ID citations [S_timestamp_hash]
       unique_citation_pattern = r'\[(S_\d+_[a-f0-9]+)\]'
       for match in re.finditer(unique_citation_pattern, answer):
           sid = match.group(1)
           if sid in src_map:
               explicit_citations.add(sid)
       
       # ...rest of the method...
   ```

4. **Citation Renumbering with Dual IDs**:
   ```python
   # Create cited sources with unique IDs and display IDs
   cited_sources = []
   renumber_map = {}
   
   for display_id, src in enumerate(cited_raw, 1):
       unique_id = src["id"]  # Keep the unique ID from the source
       display_id_str = str(display_id)  # Sequential display number
       
       # Map unique ID to display ID for answer renumbering
       renumber_map[unique_id] = display_id_str
       
       # Create entry with both unique ID and display ID
       entry = {
           "id": unique_id,  # Keep unique ID for linking
           "display_id": display_id_str,  # Add display ID for user-facing numbers
           "title": src["title"], 
           "content": src["content"],
           "parent_id": src.get("parent_id", ""),
           "is_procedural": src.get("is_procedural", False)
       }
       cited_sources.append(entry)

   # Apply display numbering to the answer text
   for unique_id, display_id in renumber_map.items():
       answer = re.sub(rf"\[{re.escape(unique_id)}\]", f"[{display_id}]", answer)
   ```

### Frontend Changes

1. **Enhanced Source Storage**:
   ```javascript
   function handleStreamingMetadata(metadata) {
     if (metadata.sources && metadata.sources.length > 0) {
       // Store sources for citation handling with unique IDs
       window.lastSources = metadata.sources.map(source => ({
         ...source,
         // Ensure we have both unique ID and display ID
         id: source.id || `source-${source.display_id || '1'}`,
         display_id: source.display_id || '1'
       }));
       
       console.log('Sources received with unique IDs:', window.lastSources.map(s => ({ id: s.id, display_id: s.display_id })));
     }
     
     // Handle other metadata like evaluation if needed
     if (metadata.evaluation) {
       console.log('Evaluation received:', metadata.evaluation);
     }
   }
   ```

2. **Enhanced Citation Click Handling**:
   ```javascript
   function handleCitationClick(citationLink) {
     const sourceId = citationLink.getAttribute('data-source-id');
     
     // Remove any existing highlights
     this.removeAllHighlights();
     
     // Get source information from the global lastSources
     if (window.lastSources && Array.isArray(window.lastSources)) {
       // Handle both numeric and unique ID citations
       let source = null;
       
       // First try to find by exact ID match (for unique IDs)
       source = window.lastSources.find(s => s.id === sourceId);
       
       // If not found and sourceId is numeric, try index-based lookup
       if (!source && /^\d+$/.test(sourceId)) {
         const sourceIndex = parseInt(sourceId) - 1;
         if (sourceIndex >= 0 && sourceIndex < window.lastSources.length) {
           source = window.lastSources[sourceIndex];
         }
       }
       
       if (source) {
         // Display source information
         // ...
       }
     }
   }
   ```

## Testing

A comprehensive test suite has been created to verify the implementation:

1. **Unique ID Generation**: Tests that unique source IDs are generated correctly and follow the expected format.
2. **Context Preparation**: Tests that `_prepare_context` uses unique IDs for sources.
3. **Citation Detection**: Tests that `_filter_cited` correctly detects both numeric and unique ID citations.
4. **Citation Renumbering**: Tests the citation renumbering logic to ensure it correctly maps unique IDs to display IDs.
5. **Cumulative Source Map**: Tests that the cumulative source map is properly maintained across conversation turns.

## Benefits

1. **Consistent Citations**: Citations now work consistently across conversation turns, even when referencing sources from previous turns.
2. **Complete Source List**: All cited sources are now included in the source list, regardless of parent document limits.
3. **Improved User Experience**: Users no longer encounter "Source information not found" errors when clicking on citations.
4. **Backward Compatibility**: The system still supports numeric citations for backward compatibility.

## Future Improvements

1. **Persistent Storage**: Consider storing the cumulative source map in a database for persistence across sessions.
2. **Source Deduplication**: Implement deduplication of sources based on content similarity to avoid redundant sources.
3. **Enhanced Citation UI**: Improve the citation UI to show more context about the source, such as its origin and relevance.
