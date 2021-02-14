#ifndef base_runner_cuh
#define base_runner_cuh

#include <functional>
#include <map>
#include <string>

typedef std::map<std::string, double> report_map;

namespace Runners
{
	class BaseRunnerParameters {};
	class BaseRunner
	{
	private:
		BaseRunnerParameters* parameters;
	};
}


#endif // !base_runner_cuh
