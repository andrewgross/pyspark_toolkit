# Changelog

## [2025-12-04] - Version 0.7.0

### Breaking Changes

- **FDTF Parameter Rename**: Renamed `output_schema` parameter to `returnType` in the `fdtf` decorator to match PySpark's `@udtf` decorator signature. Update your code: `@fdtf(output_schema="...")` becomes `@fdtf(returnType="...")`.

### Bug Fixes

- **Databricks Connect Compatibility**: Fixed cloudpickle serialization issues that caused `ModuleNotFoundError: No module named 'pyspark_toolkit'` when using `fdtf` with Databricks Connect. Helper functions and context classes are now defined locally inside the runner to avoid module reference issues during unpickling.

## [2025-12-03] - Version 0.6.0

### Features

- **Concurrent Row Processing**: FDTF decorator now supports concurrent processing with configurable thread pool using the max_workers parameter, enabling parallel execution of row transformations for improved performance. (73c4c4e, 94f54f5)
- **Resource Management**: Added init_fn and cleanup_fn parameters to FDTF for lifecycle management of resources (HTTP clients, database connections, etc.) that are initialized once per partition and properly cleaned up. (73c4c4e, 94f54f5)
- **Retry Logic**: Implemented automatic retry mechanism with configurable max_retries parameter, allowing functions to automatically retry failed operations with exponential backoff. (73c4c4e, 94f54f5)
- **Execution Metadata Tracking**: Added metadata_column parameter that automatically tracks execution attempts, timestamps, durations, and errors for each row transformation, providing visibility into processing behavior. (73c4c4e, 94f54f5)

### Enhancements

- **Flexible Return Types**: FDTF functions can now return single values, tuples, or generators - the decorator automatically normalizes all return types for consistent handling. (94f54f5)
- **Context Object Support**: FDTF functions can now optionally receive a context object (self) as the first parameter to access resources initialized by init_fn, enabling stateful row processing. (94f54f5)
- **Improved Type Signatures**: Enhanced type hints and validation for FDTF decorator parameters and function signatures for better IDE support and error messages. (94f54f5)

### Documentation

- **FDTF Examples**: Added comprehensive examples demonstrating resource management, retry logic, metadata tracking, and concurrent processing patterns to README. (6bd4b03)

### Internal

- **Test Coverage**: Expanded test suite with comprehensive coverage for concurrent execution, retry behavior, resource lifecycle, and error handling scenarios. (73c4c4e, 94f54f5)
- **Code Consolidation**: Unified CDTF and FDTF implementations under a single consistent API, simplifying the codebase while maintaining backward compatibility. (94f54f5)


## [2025-12-02] - Version 0.5.0

### Features

- **S3 Module Now Available**: The S3 presigned URL generation module is now fully functional and included in the package. Previously excluded due to heap overflow issues, the module has been fixed and is ready for production use. (f200957)
- **UDTF Helper Functions**: Added comprehensive User-Defined Table Function (UDTF) helper utilities to simplify creating and working with UDTFs in PySpark. Includes support for string-based schema definitions for improved usability. (c35c5f1, dad023c)

### Bug Fixes

- **S3 Heap Overflow**: Fixed critical heap overflow issue in S3 presigned URL generation by applying changes as individual columns rather than as a struct. This resolves the hanging behavior that previously made the module unusable. (92393a7)

### Documentation

- **UDTF/FDTF Documentation**: Added comprehensive documentation for User-Defined Table Functions (UDTF) and Flexible DataFrame Table Functions (FDTF), including examples and best practices. (bed3928)
- **S3 Examples**: Added comprehensive S3 presigned URL generation examples to README, including function reference and usage patterns. (f200957)
- **Known Issues Removed**: Removed KNOWN_ISSUES.md file as there are no longer any known issues with the toolkit. (f200957)

### Enhancements

- **String-Based Schemas**: UDTF helpers now support string-based schema definitions in addition to StructType, making schema definition more concise and readable. (dad023c)

### Internal

- **Test Coverage**: Significantly expanded test coverage for S3 module and UDTF helpers. (92393a7, c35c5f1)
- **Build Configuration**: Updated pyproject.toml to re-enable S3 module in package distribution. (f200957)

## [2025-08-06] - Version 0.4.0

### Features

- **map_concat Function**: Added custom map_concat function with right-override merge strategy for combining map/dictionary columns in DataFrames. (fa8df27)

### Bug Fixes

- **Test Infrastructure**: Fixed test configuration and added proper pytest fixtures for Spark session management, ensuring tests run reliably in parallel. (c35c5f1, da73916)

### Documentation

- **Critical Documentation Fixes**: Fixed critical errors in documentation and added comprehensive examples for key functions. (8a70585)

### Enhancements

- **Databricks Compatibility**: Removed pyspark as a direct dependency to resolve installation issues on Databricks. PySpark is now an optional dependency and users should ensure it's available in their environment. (18ac18d)
- **JSON Column Output**: Updated map_json_column to allow specifying a dedicated output column, providing more flexibility in JSON transformation workflows. (db287e3)

## [2025-08-06] - Version 0.2.0

### Breaking Changes

- **Package Rename**: Renamed package to pyspark-toolkit. Update your imports accordingly. (dae75dc)
- **Module Rename**: Renamed json module for consistency. (eec1362)
- **UUID Partitioning**: Renamed uuid partitioning function for clarity. (0abad69)

### Features

- **UUID5 Support**: Added uuid5 function for generating deterministic UUIDs from namespaces and names. (fbed2b9, 3f50791)
- **UUID Modulus**: Added UUID modulus code for consistent data partitioning based on UUID values. (08978da)
- **JSON Helpers**: Added JSON helper functions for working with JSON data in DataFrames. (267d3f3)

### Bug Fixes

- **Type Safety**: Made casting safe across PySpark 3.5 and 4.0 versions. (ddd2dd7)

### Documentation

- **README**: Added simple README with project overview and usage instructions. (e4d2c1f)
- **License**: Created LICENSE file for the project. (69dec45)

### Internal

- **S3 Module Excluded**: Temporarily excluded S3 package from distribution due to heap overflow issues. Code retained for future fixes. (daec659)
- **Modern Tooling**: Converted to modern Python tooling with uv and updated development dependencies. (0a6667d, 0c3acc9)
- **Pre-commit**: Added comprehensive pre-commit configuration for code quality. (54bec3d, 9146015)
- **Test Infrastructure**: Ensured tests have direct access to Spark session and added proper test fixtures. (a92001d)
