#ifndef data_writer_cpp
#define data_writer_cpp

#include "data_writer.hpp"


void save_image_bmp(char* file_name, int w, int h, bool* bits)
{
	char* bytes = (char*)malloc(w * h * 3);

	int byte = 0;
	for (int j = 0; j < 3 * 8 * w * h; j++)
	{
		int pow = 7 - (j % 8);
		byte += bits[j] * (1 << pow);
		if (pow == 0)
		{
			bytes[j / 8] = (char)byte;
			byte = 0;
		}
	}
	//stbi_write_bmp(file_name, w, h, 3, bytes);
	free(bytes);
}

#endif // !data_writer_cpp
