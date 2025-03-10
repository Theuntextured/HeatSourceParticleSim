#pragma once
#define DEVICE_SWITCH(RUN_ON_DEVICE, RUN_ON_HOST) if (__CUDA_ARCH__) {\
			RUN_ON_DEVICE\
		}\
else {\
	RUN_ON_HOST\
}

#ifdef _DEBUG
#define check(Condition) if(!static_cast<bool>(Condition)) __debugbreak();
#else
#define check(Condition) 
#endif