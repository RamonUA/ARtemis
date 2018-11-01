/* ===============================================================================================
Exports as a library
 ===============================================================================================*/
#ifdef ARtemis_EXPORTS
	#if defined _WIN32 || defined _WIN64
		#define ARTEMIS_LIB __declspec( dllexport )
	#else
		#define ARTEMIS_LIB
	#endif
#else
#define ARTEMIS_LIB
#endif