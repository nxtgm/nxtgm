#include "QPBO.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif
/*
template <>
	inline void QPBO<int>::get_type_information(const char*& type_name, const char*& type_format)
{
	type_name = "int";
	type_format = "d";
}

template <>
	inline void QPBO<float>::get_type_information(const char*& type_name, const char*& type_format)
{
	type_name = "float";
	type_format = "f";
}
*/
template <>
	inline void QPBO<double>::get_type_information(const char*& type_name, const char*& type_format)
{
	type_name = "double";
	type_format = "lf";
}


// Instantiations

//template class QPBO<int>;
//template class QPBO<float>;
template class QPBO<double>;
