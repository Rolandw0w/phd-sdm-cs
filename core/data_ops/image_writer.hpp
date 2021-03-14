#ifndef image_writer_h
#define image_writer_h


#ifndef INCLUDE_STB_IMAGE_WRITE_H
#define INCLUDE_STB_IMAGE_WRITE_H

#include <stdlib.h>

// if STB_IMAGE_WRITE_STATIC causes problems, try defining STBIWDEF to 'inline' or 'static inline'
#ifndef STBIWDEF
#ifdef STB_IMAGE_WRITE_STATIC
#define STBIWDEF  static
#else
#ifdef __cplusplus
#define STBIWDEF  extern "C"
#else
#define STBIWDEF  extern
#endif
#endif
#endif

#ifndef STB_IMAGE_WRITE_STATIC  // C++ forbids static forward declarations
extern int stbi_write_tga_with_rle;
extern int stbi_write_png_compression_level;
extern int stbi_write_force_png_filter;
#endif

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);

#ifdef STBI_WINDOWS_UTF8
STBIWDEF int stbiw_convert_wchar_to_utf8(char *buffer, size_t bufferlen, const wchar_t* input);
#endif
#endif

typedef void stbi_write_func(void *context, void *data, int size);

STBIWDEF void stbi_flip_vertically_on_write(int flip_boolean);

#endif//INCLUDE_STB_IMAGE_WRITE_H

#endif //image_writer_h