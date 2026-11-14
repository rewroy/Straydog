#pragma once
// The host transfers of host_io.hpp that the other units call.
uint8_t* io_slot(Device& d, size_t bytes);
void upload(Device& d, void* device, const void* host, size_t bytes);
void download(Device& d, void* host, const void* device, size_t bytes);
std::string budget_exceeded(const Device& value, uint64_t bytes);
