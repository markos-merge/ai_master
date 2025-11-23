#include <iostream>
#include <rocm_smi/rocm_smi.h>

// A helper function to check the return status of RSMI calls
void check_rsmi_status(rsmi_status_t status, const std::string &function_name) {
	if (status != RSMI_STATUS_SUCCESS) {
		const char *error_string;
		rsmi_status_string(status, &error_string);
		std::cerr << "RSMI call '" << function_name << "' failed: " << error_string
		          << std::endl;
		// Shut down and exit on failure
		rsmi_shut_down();
		exit(EXIT_FAILURE);
	}
}

int main() {
	rsmi_status_t status;

	// Initialize the ROCm SMI library
	status = rsmi_init(0);
	check_rsmi_status(status, "rsmi_init");

	// Get the number of GPU devices
	uint32_t device_count = 0;
	status = rsmi_num_monitor_devices(&device_count);
	check_rsmi_status(status, "rsmi_num_monitor_devices");

	if (device_count == 0) {
		std::cout << "Hello ROCm! - No AMD GPUs found on this system." << std::endl;
	} else {
		std::cout << "Hello ROCm! - Found " << device_count << " AMD GPU(s)."
		          << std::endl;
	}

	// Iterate over each device and print information
	for (uint32_t i = 0; i < device_count; ++i) {
		std::cout << "\n--- GPU " << i << " ---" << std::endl;

		// Get and print the GPU ID
		uint16_t device_id;
		status = rsmi_dev_id_get(i, &device_id);
		if (status == RSMI_STATUS_SUCCESS) {
			std::cout << "  Device ID: 0x" << std::hex << device_id << std::dec
			          << std::endl;
		} else {
			std::cerr << "  Could not retrieve Device ID." << std::endl;
		}

		// Get and print VRAM usage
		uint64_t vram_used = 0;
		uint64_t vram_total = 0;
		// Use the memory functions compatible with older rocm-smi-lib versions
		rsmi_status_t usage_status =
		rsmi_dev_memory_usage_get(i, RSMI_MEM_TYPE_VRAM, &vram_used);
		rsmi_status_t total_status =
		rsmi_dev_memory_total_get(i, RSMI_MEM_TYPE_VRAM, &vram_total);

		if (usage_status == RSMI_STATUS_SUCCESS &&
		    total_status == RSMI_STATUS_SUCCESS) {
			std::cout << "  VRAM Usage: " << (vram_used / (1024 * 1024)) << "MB / "
			          << (vram_total / (1024 * 1024)) << "MB" << std::endl;
		} else {
			std::cerr << "  Could not retrieve VRAM usage." << std::endl;
		}

		// Get and print the temperature (edge sensor)
		int64_t temp;
		rsmi_temperature_metric_t metric =
		RSMI_TEMP_CURRENT; // Use the enum compatible with your library
		rsmi_temperature_type_t type = RSMI_TEMP_TYPE_EDGE;
		status = rsmi_dev_temp_metric_get(i, type, metric, &temp);
		if (status == RSMI_STATUS_SUCCESS) {
			// Temperature is returned in millidegrees Celsius
			std::cout << "  Temperature: " << (temp / 1000.0) << " C" << std::endl;
		} else {
			std::cerr << "  Could not retrieve temperature." << std::endl;
		}
	}

	// Shut down the ROCm SMI library
	status = rsmi_shut_down();
	check_rsmi_status(status, "rsmi_shut_down");
	std::cout << "\nGoodbye ROCm!" << std::endl;

	return 0;
}
